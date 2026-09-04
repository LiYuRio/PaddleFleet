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

"""`liger_ce_multimax_tuning` selects a different multimax CE kernel body *and* a
different launch geometry, and it is not bit-identical to the default. This file
pins down the two things that can be asserted -- that the flag is off by default
and that it really switches both -- and reports the numerics rather than
asserting a tolerance nobody has justified.

Why the body and the geometry cannot be tested separately: num_warps 4 -> 1 is
what removes the shared-memory barriers from the eight param-grad reductions,
and the body rewrite (one-hot correction hoisted out of the loop, unmasked main
loop, `exp()*inv_d`) is what keeps the 1024-wide tile spill-free. Selecting one
without the other is not a configuration anyone measured.
"""

import unittest
from unittest import mock

import paddle

from paddlefleet.triton_ops.fused_linear_cross_entropy import (
    fused_linear_cross_entropy as flce,
)

BT, H, V = 256, 64, 2048
IGNORE = -100


def _inputs(seed=20260904):
    paddle.seed(seed)
    x = paddle.randn([BT, H], dtype="bfloat16")
    w = paddle.randn([V, H], dtype="bfloat16") * 0.02
    # The launcher only writes grad_input / grad_weight when the operands ask
    # for it, and those are the tensors the kernel body actually differs on.
    x.stop_gradient = False
    w.stop_gradient = False
    t = paddle.randint(0, V, [BT], dtype="int64")
    # SegLU knobs: four ranges and four scales, as GPTLMHead emits them.
    ranges = paddle.to_tensor([-1.0, 1.0, -2.0, 2.0], dtype="float32")
    ts = paddle.to_tensor([0.1, 0.2, 0.05, 0.05], dtype="float32")
    ranges.stop_gradient = False
    ts.stop_gradient = False
    return x, w, t, ranges, ts


class _LaunchSpy:
    """Records ``kernel[grid]`` and still launches the real kernel."""

    def __init__(self, real):
        self._real = real
        self.launches = 0

    def __getitem__(self, grid):
        self.launches += 1
        return self._real[grid]


def _forward(tuning, freeze_params=False):
    x, w, t, ranges, ts = _inputs()
    if freeze_params:
        ranges.stop_gradient = True
        ts.stop_gradient = True
    return flce.fused_linear_cross_entropy_forward(
        _input=x,
        weight=w,
        target=t,
        bias=None,
        ignore_index=IGNORE,
        reduction="none",
        num_chunks=1,
        multimax_ranges=ranges,
        multimax_ts=ts,
        liger_ce_multimax_tuning=tuning,
    )


def _max_rel(a, b):
    a = a.astype("float64")
    b = b.astype("float64")
    return float(((a - b).abs() / b.abs().clip(min=1e-30)).max())


class TestLigerCEMultimaxTuning(unittest.TestCase):
    def test_config_field_defaults_off(self):
        from types import SimpleNamespace

        from paddlefleet.transformer.transformer_config import (
            TransformerConfig,
        )

        base = {"hidden_size": 8, "num_attention_heads": 2}
        self.assertFalse(
            TransformerConfig.from_config(
                SimpleNamespace(**base)
            ).liger_ce_multimax_tuning
        )
        self.assertTrue(
            TransformerConfig.from_config(
                SimpleNamespace(liger_ce_multimax_tuning=True, **base)
            ).liger_ce_multimax_tuning
        )

    def _launch_counts(self, tuning):
        old = _LaunchSpy(flce.liger_cross_entropy_multimax_kernel)
        new = _LaunchSpy(flce.liger_cross_entropy_multimax_tuned_kernel)
        with (
            mock.patch.object(flce, "liger_cross_entropy_multimax_kernel", old),
            mock.patch.object(
                flce, "liger_cross_entropy_multimax_tuned_kernel", new
            ),
        ):
            _forward(tuning=tuning)
        return old.launches, new.launches

    def test_default_uses_the_original_kernel(self):
        old, new = self._launch_counts(tuning=False)
        self.assertGreater(old, 0, "default did not run the original kernel")
        self.assertEqual(new, 0, "default reached the tuned kernel")

    def test_flag_selects_the_tuned_kernel(self):
        old, new = self._launch_counts(tuning=True)
        self.assertGreater(new, 0, "flag did not reach the tuned kernel")
        self.assertEqual(old, 0, "flag still ran the original kernel")

    def test_launch_geometry_has_two_tiers(self):
        # num_warps pinned to 1; the tile cap depends on whether this call has
        # to reduce the eight param grads.
        self.assertEqual(
            flce._select_ce_multimax_launch_config(V, has_param_grads=True),
            (flce.CE_MULTIMAX_BLOCK_SIZE_CAP, 1),
        )
        self.assertEqual(
            flce._select_ce_multimax_launch_config(V, has_param_grads=False),
            (flce.CE_MULTIMAX_BLOCK_SIZE_CAP_NO_PARAM_GRAD, 1),
        )
        # A vocab smaller than the cap must not be rounded up past itself.
        self.assertEqual(flce._select_ce_multimax_launch_config(256)[0], 256)

    def test_numerics_against_the_default_are_reported(self):
        for freeze in (False, True):
            ref = _forward(tuning=False, freeze_params=freeze)
            got = _forward(tuning=True, freeze_params=freeze)
            self.assertEqual(len(got), len(ref))
            names = ("loss", "grad_input", "grad_weight", "grad_bias")
            if len(ref) > 4:
                names = (*names, "grad_mm_ranges", "grad_mm_ts")
            for name, a, b in zip(names, got, ref, strict=True):
                if a is None or b is None:
                    self.assertIs(a, b, f"{name}: one side is None")
                    continue
                self.assertTrue(
                    bool(paddle.isfinite(a.astype("float32")).all()),
                    f"{name} is not finite with the flag on",
                )
                same = bool(
                    paddle.equal_all(a.astype("float32"), b.astype("float32"))
                )
                print(
                    f"[liger_ce] freeze_params={freeze} {name}: "
                    f"bit-identical={same} max_rel={_max_rel(a, b):.3e}"
                )


if __name__ == "__main__":
    unittest.main()
