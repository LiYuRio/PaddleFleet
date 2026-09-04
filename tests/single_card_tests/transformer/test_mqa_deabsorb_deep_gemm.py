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

"""`MQALatentAttention._deabsorb` forwards `grouped_matmul_deep_gemm` down to
`fused_grouped_matmul`. It is a staticmethod on the split-`kv_b` branch, so the
flag has to be passed in rather than read off `self`; these tests cover that
branch directly instead of standing up the whole attention module.
"""

import unittest

import paddle

from paddlefleet.transformer.mqa_latent_attention import MQALatentAttention

B, S, H, KV_LORA, V_HEAD = 2, 4, 8, 32, 16


def _inputs(seed=20260904):
    paddle.seed(seed)
    core_out = paddle.randn([B, S, H * KV_LORA], dtype="bfloat16")
    # split_kv_b layout: [h, v_head_dim, kv_lora_rank], the [G, R, D] contract.
    v_b = paddle.randn([H, V_HEAD, KV_LORA], dtype="bfloat16") * 0.02
    return core_out, v_b


class TestDeabsorbDeepGemmForwarding(unittest.TestCase):
    def test_split_kv_b_runs_with_the_flag_off_and_on(self):
        core_out, v_b = _inputs()
        off = MQALatentAttention._deabsorb(
            core_out, v_b, split_kv_b=True, use_deep_gemm=False
        )
        on = MQALatentAttention._deabsorb(
            core_out, v_b, split_kv_b=True, use_deep_gemm=True
        )

        self.assertEqual(list(off.shape), [B, S, H * V_HEAD])
        self.assertEqual(list(on.shape), list(off.shape))
        self.assertEqual(on.dtype, off.dtype)
        self.assertTrue(bool(paddle.isfinite(on.astype("float32")).all()))

    def test_flag_defaults_off(self):
        # The default has to match `fused_grouped_matmul`'s own default, so a
        # caller that has not been taught about the flag keeps the Triton path.
        import inspect

        sig = inspect.signature(MQALatentAttention._deabsorb)
        self.assertIs(sig.parameters["use_deep_gemm"].default, False)

    def test_flag_reaches_fused_grouped_matmul(self):
        from unittest import mock

        from paddlefleet import triton_ops

        seen = []
        real = triton_ops.fused_grouped_matmul

        def spy(*args, **kwargs):
            seen.append(kwargs.get("use_deep_gemm"))
            return real(*args, **kwargs)

        core_out, v_b = _inputs()
        with mock.patch.object(triton_ops, "fused_grouped_matmul", spy):
            for flag in (False, True):
                MQALatentAttention._deabsorb(
                    core_out, v_b, split_kv_b=True, use_deep_gemm=flag
                )
        self.assertEqual(seen, [False, True])


if __name__ == "__main__":
    unittest.main()
