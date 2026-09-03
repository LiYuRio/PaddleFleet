# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""`grouped_matmul_fusion` carries two separate changes with different contracts.

The tuned Triton meta params (`_TUNED` / `_tuned`) are claimed bit-identical to
the shipped 128x128x64 defaults, and are therefore unconditional. That claim is
checked here by running the same call with `_tuned` neutered, which is exactly
the pre-change configuration, and comparing out / dx / dw bit-for-bit.

The DeepGEMM path (`_dg_run`) hands the forward and dx GEMMs to another library,
so it is opt-in via `use_deep_gemm` / `config.grouped_matmul_deep_gemm`. What is
asserted about it is the *plumbing*: off by default, never reached unless asked,
and dw stays on Triton either way. Its numerics are reported rather than
asserted -- see `test_deep_gemm_numerics_are_reported`, which prints what it
measured instead of encoding a tolerance nobody has justified.
"""

import unittest
from unittest import mock

import paddle

from paddlefleet.triton_ops import (
    fused_grouped_matmul,
    grouped_matmul_fusion as gm,
)

# G, M, R, D. R and D are multiples of 256 so `_tuned` actually applies to all
# three kernels (fwd tiles M x R, dx tiles M x D, dw tiles R x D).
G, M, R, D = 4, 512, 256, 256


def _inputs(seed, dtype="bfloat16"):
    paddle.seed(seed)
    x = paddle.randn([M, G, D], dtype=dtype)
    w = paddle.randn([G, R, D], dtype=dtype)
    dy = paddle.randn([M, G, R], dtype=dtype)
    return x, w, dy


def _run(x, w, dy, **kwargs):
    """out, dx, dw for one configuration."""
    xv, wv = x.detach(), w.detach()
    xv.stop_gradient = False
    wv.stop_gradient = False
    out = fused_grouped_matmul(xv, wv, **kwargs)
    out.backward(dy)
    return out, xv.grad, wv.grad


def _bit_equal(a, b):
    """``equal_all`` has no bfloat16 kernel; widening to fp32 is exact."""
    return bool(paddle.equal_all(a.astype("float32"), b.astype("float32")))


def _max_rel(a, b):
    a = a.astype("float64")
    b = b.astype("float64")
    return float(((a - b).abs() / b.abs().clip(min=1e-30)).max())


class TestTunedMetaParams(unittest.TestCase):
    """The unconditional half: must be bit-identical to the old defaults."""

    def test_tuned_launch_is_bit_exact_against_the_shipped_defaults(self):
        x, w, dy = _inputs(20260903)

        # `_tuned` returning {} makes every launcher fall back to the kernel
        # signature defaults, i.e. 128x128x64 / 4 warps / 3 stages.
        with mock.patch.object(gm, "_tuned", lambda *a, **k: {}):
            ref = _run(x, w, dy)
        got = _run(x, w, dy)

        for name, a, b in zip(("out", "dx", "dw"), got, ref, strict=True):
            with self.subTest(tensor=name):
                self.assertTrue(
                    _bit_equal(a, b), f"{name} differs from the old meta params"
                )

    def test_tuned_declines_shapes_that_do_not_tile_exactly(self):
        # The sweep covers one shape; a 256-wide tile on a smaller R or D would
        # compute mostly masked lanes, so anything that is not an exact fit has
        # to keep the shipped defaults rather than an extrapolation.
        self.assertEqual(gm._tuned("fwd", 512, 128), {})
        self.assertEqual(gm._tuned("dw", 200, 256), {})
        self.assertEqual(gm._tuned("fwd", 512, 256), gm._TUNED["fwd"])
        self.assertEqual(gm._tuned("dw", 256, 512), gm._TUNED["dw"])


class TestDeepGemmIsOptIn(unittest.TestCase):
    """The gated half: the plumbing, not the numerics."""

    def test_config_field_defaults_off(self):
        from types import SimpleNamespace

        from paddlefleet.transformer.transformer_config import (
            TransformerConfig,
        )

        base = {"hidden_size": 8, "num_attention_heads": 2}
        cfg = TransformerConfig.from_config(SimpleNamespace(**base))
        self.assertFalse(cfg.grouped_matmul_deep_gemm)
        on = TransformerConfig.from_config(
            SimpleNamespace(grouped_matmul_deep_gemm=True, **base)
        )
        self.assertTrue(on.grouped_matmul_deep_gemm)

    def test_default_never_reaches_deep_gemm(self):
        x, w, dy = _inputs(1)
        with mock.patch.object(gm, "_dg_run", wraps=gm._dg_run) as dg:
            _run(x, w, dy)
        dg.assert_not_called()

    def test_opt_in_asks_for_it_in_both_forward_and_dx(self):
        x, w, dy = _inputs(2)
        with mock.patch.object(gm, "_dg_run", wraps=gm._dg_run) as dg:
            _run(x, w, dy, use_deep_gemm=True)
        asked = [c.args[0] for c in dg.call_args_list]
        self.assertEqual(asked, ["fwd", "dx"])

    def test_opt_in_does_not_touch_dw(self):
        # bf16 DeepGEMM has no "bhd,bhr->hdr" expression, so dw must be
        # bit-identical whether or not the flag is set.
        x, w, dy = _inputs(3)
        off = _run(x, w, dy)
        on = _run(x, w, dy, use_deep_gemm=True)
        self.assertTrue(_bit_equal(on[2], off[2]), "dw changed with the flag")

    def test_fp16_falls_back(self):
        # einsum.hpp asserts bf16 on all three operands, so fp16 must decline
        # rather than reach it.
        x, w, dy = _inputs(4, dtype="float16")
        ref = _run(x, w, dy)
        got = _run(x, w, dy, use_deep_gemm=True)
        for name, a, b in zip(("out", "dx", "dw"), got, ref, strict=True):
            with self.subTest(tensor=name):
                self.assertTrue(_bit_equal(a, b))

    def test_deep_gemm_numerics_are_reported(self):
        if gm._dg_einsum() is None:
            self.skipTest("DeepGEMM einsum unavailable")
        x, w, dy = _inputs(5)
        ref = _run(x, w, dy)
        got = _run(x, w, dy, use_deep_gemm=True)
        for name, a, b in zip(("out", "dx"), got[:2], ref[:2], strict=True):
            same = _bit_equal(a, b)
            print(
                f"[deep_gemm] G={G} M={M} R={R} D={D} {name}: "
                f"bit-identical={same} max_rel={_max_rel(a, b):.3e}"
            )
        # dw is the one thing the flag must not move.
        self.assertTrue(_bit_equal(got[2], ref[2]))


if __name__ == "__main__":
    unittest.main()
