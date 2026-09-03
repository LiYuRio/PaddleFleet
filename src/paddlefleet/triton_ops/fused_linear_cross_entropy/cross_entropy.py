#!/usr/bin/env python3

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

"""
Fused Cross Entropy Triton Kernel。

基于在线 Softmax 算法，在单次扫描中同时完成 loss 计算和梯度计算，
避免 Paddle 原生实现中保存完整 softmax 中间张量带来的显存开销。

Multimax variant (`liger_cross_entropy_multimax_kernel`) additionally fuses
the learnable SegLU-style segmented modulation
    SegLU(x) = x + t0·max(r0-x,0) + t1·max(x-r1,0)
                + t2·max(r2-x,0)^2 + t3·max(x-r3,0)^2
into the same kernel: SegLU is applied in registers during both the lse pass
and the grad-write pass; per-row partial sums for grad_ranges/grad_ts are
accumulated in registers and atomic-added to global [4]-shape fp32 buffers.
This avoids materializing the four ReLU intermediates and SegLU output as
separate tensors, dropping per-chunk peak memory back to ~1× [C, V].
"""

import triton
import triton.language as tl

from ..triton_compat import enable_compat_on_triton_kernel


@enable_compat_on_triton_kernel
@triton.jit
def liger_cross_entropy_kernel(  # pragma: no cover - triton kernel body compiles to PTX, not python-instrumentable
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore,
    ignore_index,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_GRADIENTS: tl.constexpr,
):
    """计算交叉熵 loss，并可选地原地写回梯度。"""
    program_id = tl.program_id(0).to(tl.int64)

    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    X_ptr += program_id * X_stride

    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    loss_ptr += program_id * loss_stride

    m = float("-inf")
    d = 0.0
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
        ).cast(tl.float32)
        block_max = tl.max(X_block)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    lse = m + tl.log(d)

    if HAS_GRADIENTS:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            X_block = tl.load(
                X_ptr + X_offsets,
                mask=X_offsets < n_cols,
                other=float("-inf"),
            ).cast(tl.float32)

            X_block = tl.exp(X_block - m) / d
            X_block = tl.where(X_offsets != y, X_block, X_block - 1.0)

            if reduction == "mean":
                X_block = X_block / n_non_ignore

            tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)

    tl.debug_barrier()

    loss = lse - ori_X_y

    if reduction == "mean":
        loss = loss / n_non_ignore

    tl.store(loss_ptr, loss)


@enable_compat_on_triton_kernel
@triton.jit
def liger_cross_entropy_multimax_kernel(  # pragma: no cover - triton kernel body compiles to PTX, not python-instrumentable
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore,
    ignore_index,
    # Multimax learnable params, passed as fp32 scalars (compiled in once
    # per chunk; ranges/ts are tiny [4] tensors so the host->device sync
    # to extract them is negligible vs the chunk's GEMM/CE cost).
    r0,
    r1,
    r2,
    r3,
    t0,
    t1,
    t2,
    t3,
    # Per-chunk fp32 [4] grad accumulators for ranges/ts.
    grad_r_ptr,
    grad_t_ptr,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_GRADIENTS: tl.constexpr,
    HAS_MULTIMAX_GRADIENTS: tl.constexpr,
):
    """Liger CE kernel with fused SegLU activation + closed-form SegLU backward.

    Forward:  L = lse(SegLU(X)) - SegLU(X)[y]   (per row)
    Backward: writes grad_x = dL/dX in place into X_ptr; atomic-adds the
              per-row partial sums for grad_ranges and grad_ts into the
              fp32 [4] global buffers grad_r_ptr / grad_t_ptr.

    SegLU is computed in registers from the loaded X block, so no extra HBM
    traffic relative to the no-multimax kernel.
    """
    program_id = tl.program_id(0).to(tl.int64)

    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    X_ptr += program_id * X_stride

    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    loss_ptr += program_id * loss_stride

    # Apply SegLU to the y-th element (loss target) once.
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)
    my0 = tl.maximum(r0 - ori_X_y, 0.0)
    my1 = tl.maximum(ori_X_y - r1, 0.0)
    my2 = tl.maximum(r2 - ori_X_y, 0.0)
    my3 = tl.maximum(ori_X_y - r3, 0.0)
    seglu_X_y = ori_X_y + t0 * my0 + t1 * my1 + t2 * my2 * my2 + t3 * my3 * my3

    # Pass 1: online lse over SegLU(X).
    m = float("-inf")
    d = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        in_bounds = X_offsets < n_cols
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=in_bounds,
            other=0.0,  # finite filler; we re-mask SegLU output to -inf below
        ).cast(tl.float32)
        m0_b = tl.maximum(r0 - X_block, 0.0)
        m1_b = tl.maximum(X_block - r1, 0.0)
        m2_b = tl.maximum(r2 - X_block, 0.0)
        m3_b = tl.maximum(X_block - r3, 0.0)
        seglu_b = (
            X_block
            + t0 * m0_b
            + t1 * m1_b
            + t2 * m2_b * m2_b
            + t3 * m3_b * m3_b
        )
        # Padded lanes contribute -inf to lse so they vanish in exp().
        seglu_b = tl.where(in_bounds, seglu_b, float("-inf"))
        block_max = tl.max(seglu_b)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(seglu_b - m_new))
        m = m_new

    lse = m + tl.log(d)

    if HAS_GRADIENTS or HAS_MULTIMAX_GRADIENTS:
        # Per-row scalar accumulators for grad_ranges and grad_ts.
        gr0 = 0.0
        gr1 = 0.0
        gr2 = 0.0
        gr3 = 0.0
        gt0 = 0.0
        gt1 = 0.0
        gt2 = 0.0
        gt3 = 0.0

        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            in_bounds = X_offsets < n_cols
            X_block = tl.load(
                X_ptr + X_offsets,
                mask=in_bounds,
                other=0.0,
            ).cast(tl.float32)
            # Recompute SegLU forward in registers (free vs HBM reload).
            m0_b = tl.maximum(r0 - X_block, 0.0)
            m1_b = tl.maximum(X_block - r1, 0.0)
            m2_b = tl.maximum(r2 - X_block, 0.0)
            m3_b = tl.maximum(X_block - r3, 0.0)
            seglu_b = (
                X_block
                + t0 * m0_b
                + t1 * m1_b
                + t2 * m2_b * m2_b
                + t3 * m3_b * m3_b
            )
            # Softmax over SegLU output, then subtract one-hot at target.
            grad_out = tl.exp(seglu_b - m) / d
            grad_out = tl.where(X_offsets != y, grad_out, grad_out - 1.0)
            if reduction == "mean":
                grad_out = grad_out / n_non_ignore
            # Padded lanes must contribute zero to grads (both stored grad
            # and the param-grad reductions).
            grad_out = tl.where(in_bounds, grad_out, 0.0)

            if HAS_GRADIENTS:
                # SegLU backward: d_out/d_x = 1 - t0*1{r0>x} + t1*1{x>r1}
                #                              - 2*t2*max(r2-x,0) + 2*t3*max(x-r3,0)
                mask0 = (m0_b > 0.0).to(tl.float32)
                mask1 = (m1_b > 0.0).to(tl.float32)
                dx_dx = (
                    1.0
                    - t0 * mask0
                    + t1 * mask1
                    - 2.0 * t2 * m2_b
                    + 2.0 * t3 * m3_b
                )
                grad_x = grad_out * dx_dx
                tl.store(X_ptr + X_offsets, grad_x, mask=in_bounds)

            if HAS_MULTIMAX_GRADIENTS:
                # Per-row partial sums for grad_ts and grad_ranges.
                #   d_out/d_t0 = m0,   d_out/d_t1 = m1
                #   d_out/d_t2 = m2^2, d_out/d_t3 = m3^2
                #   d_out/d_r0 =  t0*1{r0>x},  d_out/d_r1 = -t1*1{x>r1}
                #   d_out/d_r2 =  2*t2*m2,     d_out/d_r3 = -2*t3*m3
                mm0 = (m0_b > 0.0).to(tl.float32)
                mm1 = (m1_b > 0.0).to(tl.float32)
                gt0 += tl.sum(grad_out * m0_b)
                gt1 += tl.sum(grad_out * m1_b)
                gt2 += tl.sum(grad_out * m2_b * m2_b)
                gt3 += tl.sum(grad_out * m3_b * m3_b)
                gr0 += t0 * tl.sum(grad_out * mm0)
                gr1 += -t1 * tl.sum(grad_out * mm1)
                gr2 += 2.0 * t2 * tl.sum(grad_out * m2_b)
                gr3 += -2.0 * t3 * tl.sum(grad_out * m3_b)

        if HAS_MULTIMAX_GRADIENTS:
            # One atomic_add per row per scalar (8 total). Race-free across
            # programs and well below kernel runtime.
            tl.atomic_add(grad_r_ptr + 0, gr0)
            tl.atomic_add(grad_r_ptr + 1, gr1)
            tl.atomic_add(grad_r_ptr + 2, gr2)
            tl.atomic_add(grad_r_ptr + 3, gr3)
            tl.atomic_add(grad_t_ptr + 0, gt0)
            tl.atomic_add(grad_t_ptr + 1, gt1)
            tl.atomic_add(grad_t_ptr + 2, gt2)
            tl.atomic_add(grad_t_ptr + 3, gt3)

    tl.debug_barrier()

    loss = lse - seglu_X_y

    if reduction == "mean":
        loss = loss / n_non_ignore

    tl.store(loss_ptr, loss)


# ---------------------------------------------------------------------------
# Tuned variant of the multimax kernel, selected by
# ``config.liger_ce_multimax_tuning``. The kernel above is left exactly as it
# was: this one is not bit-identical to it (see the notes in its docstring),
# so the flag has to be able to choose, not just re-tile.
# ---------------------------------------------------------------------------
@triton.jit
def _seglu_fwd(  # pragma: no cover - triton
    x, r0, r1, r2, r3, t0, t1, t2, t3
):
    """SegLU 前向 + 四个 ReLU 中间量（反向和参数梯度都要用）。

    SegLU(x) = x + t0·max(r0-x,0) + t1·max(x-r1,0)
                 + t2·max(r2-x,0)^2 + t3·max(x-r3,0)^2

    m2sq/m3sq 单独返回：`t2*m2*m2` 与 `grad_out*m2*m2` 里的平方是同一个量，
    显式共用能省掉一条 FMUL/元素（原写法两处各算一次，编译器因为结合顺序
    不同无法 CSE）。
    """
    m0 = tl.maximum(r0 - x, 0.0)
    m1 = tl.maximum(x - r1, 0.0)
    m2 = tl.maximum(r2 - x, 0.0)
    m3 = tl.maximum(x - r3, 0.0)
    m2sq = m2 * m2
    m3sq = m3 * m3
    s = x + t0 * m0 + t1 * m1 + t2 * m2sq + t3 * m3sq
    return s, m0, m1, m2, m3, m2sq, m3sq


@triton.jit
def _mm_lse_block(  # pragma: no cover - triton
    X_ptr,
    offsets,
    n_cols,
    MASKED: tl.constexpr,
    m,
    d,
    r0,
    r1,
    r2,
    r3,
    t0,
    t1,
    t2,
    t3,
):
    """pass 1 的一块：在线 max / 加和。

    MASKED=False 时连 `offsets < n_cols` 的比较和 `tl.where` 都不生成
    —— 整行里只有最后一块可能是残块，让 98%+ 的迭代不为它付钱。
    """
    if MASKED:
        mask = offsets < n_cols
        x = tl.load(X_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
    else:
        x = tl.load(X_ptr + offsets).cast(tl.float32)
    s, _m0, _m1, _m2, _m3, _q2, _q3 = _seglu_fwd(
        x, r0, r1, r2, r3, t0, t1, t2, t3
    )
    if MASKED:
        # 补位 lane 贡献 -inf，exp() 里自然消失。
        s = tl.where(mask, s, float("-inf"))
    m_new = tl.maximum(m, tl.max(s))
    d = d * tl.exp(m - m_new) + tl.sum(tl.exp(s - m_new))
    return m_new, d


@triton.jit
def _mm_grad_block(  # pragma: no cover - triton
    X_ptr,
    offsets,
    n_cols,
    MASKED: tl.constexpr,
    m,
    inv_d,
    n_non_ignore,
    r0,
    r1,
    r2,
    r3,
    t0,
    t1,
    t2,
    t3,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    reduction: tl.constexpr,
    HAS_GRADIENTS: tl.constexpr,
    HAS_MULTIMAX_GRADIENTS: tl.constexpr,
):
    """pass 2 的一块：softmax-grad + SegLU 反向 + 八个参数梯度的行内偏和。

    注意这里 **不做 one-hot 减 1**：目标列 y 的修正是闭式标量，由 kernel 在
    循环外一次性补上（见 `liger_cross_entropy_multimax_kernel`）。因此循环
    体里没有 `offsets != y` 的比较，也没有对应的 select。

    a0..a3 = Σ g·m0 / m1 / m2² / m3²         （grad_ts 的原始和）
    a4..a7 = Σ g·1{m0>0} / 1{m1>0} / m2 / m3 （grad_ranges 的原始和，
                                              t0/-t1/2t2/-2t3 的缩放放在循环外）
    """
    if MASKED:
        mask = offsets < n_cols
        x = tl.load(X_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
    else:
        x = tl.load(X_ptr + offsets).cast(tl.float32)
    s, m0, m1, m2, m3, m2sq, m3sq = _seglu_fwd(
        x, r0, r1, r2, r3, t0, t1, t2, t3
    )
    # inv_d 在循环外算一次：`exp(...)/d` 里的 IEEE 除法即使 d 是循环不变量，
    # 也要留 1 FMUL + 2 FFMA/元素做 Newton 修正；换成乘倒数只剩 1 FMUL。
    g = tl.exp(s - m) * inv_d
    if reduction == "mean":
        g = g / n_non_ignore
    if MASKED:
        # 补位 lane 对存下的梯度和参数梯度都必须贡献 0。
        g = tl.where(mask, g, 0.0)

    p0 = m0 > 0.0
    p1 = m1 > 0.0
    if HAS_GRADIENTS:
        # d_out/d_x = 1 - t0·1{r0>x} + t1·1{x>r1}
        #               - 2·t2·max(r2-x,0) + 2·t3·max(x-r3,0)
        # 前两项直接 select 常数，省掉「谓词转 0/1 fp32 再 FFMA」这两步。
        c = tl.where(p0, 1.0 - t0, 1.0)
        c = tl.where(p1, c + t1, c)
        dx_dx = c - 2.0 * t2 * m2 + 2.0 * t3 * m3
        if MASKED:
            tl.store(X_ptr + offsets, g * dx_dx, mask=mask)
        else:
            tl.store(X_ptr + offsets, g * dx_dx)

    if HAS_MULTIMAX_GRADIENTS:
        #   d_out/d_t0 = m0,   d_out/d_t1 = m1
        #   d_out/d_t2 = m2^2, d_out/d_t3 = m3^2
        #   d_out/d_r0 =  t0·1{r0>x},  d_out/d_r1 = -t1·1{x>r1}
        #   d_out/d_r2 =  2·t2·m2,     d_out/d_r3 = -2·t3·m3
        a0 += tl.sum(g * m0)
        a1 += tl.sum(g * m1)
        a2 += tl.sum(g * m2sq)
        a3 += tl.sum(g * m3sq)
        a4 += tl.sum(tl.where(p0, g, 0.0))
        a5 += tl.sum(tl.where(p1, g, 0.0))
        a6 += tl.sum(g * m2)
        a7 += tl.sum(g * m3)
    return a0, a1, a2, a3, a4, a5, a6, a7


@enable_compat_on_triton_kernel
@triton.jit
def liger_cross_entropy_multimax_tuned_kernel(  # pragma: no cover - triton kernel body compiles to PTX, not python-instrumentable
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore,
    ignore_index,
    # Multimax learnable params, passed as fp32 scalars (compiled in once
    # per chunk; ranges/ts are tiny [4] tensors so the host->device sync
    # to extract them is negligible vs the chunk's GEMM/CE cost).
    r0,
    r1,
    r2,
    r3,
    t0,
    t1,
    t2,
    t3,
    # Per-chunk fp32 [4] grad accumulators for ranges/ts.
    grad_r_ptr,
    grad_t_ptr,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_GRADIENTS: tl.constexpr,
    HAS_MULTIMAX_GRADIENTS: tl.constexpr,
):
    """Liger CE kernel with fused SegLU activation + closed-form SegLU backward.

    Forward:  L = lse(SegLU(X)) - SegLU(X)[y]   (per row)
    Backward: writes grad_x = dL/dX in place into X_ptr; atomic-adds the
              per-row partial sums for grad_ranges and grad_ts into the
              fp32 [4] global buffers grad_r_ptr / grad_t_ptr.

    SegLU is computed in registers from the loaded X block, so no extra HBM
    traffic relative to the no-multimax kernel.

    实测（B30Z, BT=32768, V=201216, bf16, num_chunks=1）：这个 kernel 是**指令
    发射受限**，不是 DRAM 受限（dram_r+dram_w ≈ 17.7%，issue ≈ 68%）。SASS 逐块
    统计：pass 2 的循环体 66 条指令/元素、pass 1 30 条/元素，全 grid 20.0 G
    warp-指令，发射地板 16.6 ms vs 实测 25.1 ms。所以这里的写法都是为了**少发
    指令 + 少占寄存器**（寄存器决定 block/SM，直接决定发射效率）：
      1) one-hot 修正搬到循环外（下面的 `补上目标列 y`）—— 循环体里不再有
         `offsets != y` 的比较/select，活跃区间也短了，寄存器 168 -> 79；
      2) `exp(..)/d` 换成 `exp(..)*inv_d`；
      3) 主循环整块无 mask，只有尾块带 mask；
      4) `launch config 用 (BLOCK_SIZE<=1024, num_warps=1)`
         （见 fused_linear_cross_entropy._select_ce_multimax_launch_config）：
         一个 block 就是一个 warp，`tl.sum`/`tl.max` 退化成纯 warp shuffle，
         BAR.SYNC 和 shared memory 全部消失（原 2048/4 配置下 pass 2 每次迭代
         24 条 BAR.SYNC）。
    合计 25.13 ms -> 13.64 ms（x1.84），寄存器 128 -> 70，spill 仍为 0，
    可驻留线程 512/SM -> 896/SM。
    """
    program_id = tl.program_id(0).to(tl.int64)

    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    X_ptr += program_id * X_stride

    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    loss_ptr += program_id * loss_stride

    # Apply SegLU to the y-th element (loss target) once. 这几个标量在循环外
    # 复用两次：算 loss，以及给 8 个参数梯度补 one-hot 修正。
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)
    seglu_X_y, my0, my1, my2, my3, my2sq, my3sq = _seglu_fwd(
        ori_X_y, r0, r1, r2, r3, t0, t1, t2, t3
    )

    # 整块 / 残块的分界。n_cols 是 BLOCK_SIZE 整数倍时 n_main == n_cols，
    # 残块整支被跳过。
    n_main = (n_cols // BLOCK_SIZE) * BLOCK_SIZE

    # Pass 1: online lse over SegLU(X).
    m = float("-inf")
    d = 0.0
    for i in range(0, n_main, BLOCK_SIZE):
        m, d = _mm_lse_block(
            X_ptr,
            i + tl.arange(0, BLOCK_SIZE),
            n_cols,
            False,
            m,
            d,
            r0,
            r1,
            r2,
            r3,
            t0,
            t1,
            t2,
            t3,
        )
    if n_main < n_cols:
        m, d = _mm_lse_block(
            X_ptr,
            n_main + tl.arange(0, BLOCK_SIZE),
            n_cols,
            True,
            m,
            d,
            r0,
            r1,
            r2,
            r3,
            t0,
            t1,
            t2,
            t3,
        )

    lse = m + tl.log(d)

    if HAS_GRADIENTS or HAS_MULTIMAX_GRADIENTS:
        inv_d = 1.0 / d
        # Per-row scalar accumulators for grad_ranges and grad_ts.
        a0 = 0.0
        a1 = 0.0
        a2 = 0.0
        a3 = 0.0
        a4 = 0.0
        a5 = 0.0
        a6 = 0.0
        a7 = 0.0

        for i in range(0, n_main, BLOCK_SIZE):
            a0, a1, a2, a3, a4, a5, a6, a7 = _mm_grad_block(
                X_ptr,
                i + tl.arange(0, BLOCK_SIZE),
                n_cols,
                False,
                m,
                inv_d,
                n_non_ignore,
                r0,
                r1,
                r2,
                r3,
                t0,
                t1,
                t2,
                t3,
                a0,
                a1,
                a2,
                a3,
                a4,
                a5,
                a6,
                a7,
                reduction,
                HAS_GRADIENTS,
                HAS_MULTIMAX_GRADIENTS,
            )
        if n_main < n_cols:
            a0, a1, a2, a3, a4, a5, a6, a7 = _mm_grad_block(
                X_ptr,
                n_main + tl.arange(0, BLOCK_SIZE),
                n_cols,
                True,
                m,
                inv_d,
                n_non_ignore,
                r0,
                r1,
                r2,
                r3,
                t0,
                t1,
                t2,
                t3,
                a0,
                a1,
                a2,
                a3,
                a4,
                a5,
                a6,
                a7,
                reduction,
                HAS_GRADIENTS,
                HAS_MULTIMAX_GRADIENTS,
            )

        # 补上目标列 y：循环里用的是 p_y，真正的 grad_out[y] 是 p_y - 1
        # （mean 下 (p_y-1)/N）。差值 -1 对八个和的贡献是闭式标量。
        if reduction == "mean":
            w = -1.0 / n_non_ignore
        else:
            w = -1.0
        if HAS_MULTIMAX_GRADIENTS:
            a0 += w * my0
            a1 += w * my1
            a2 += w * my2sq
            a3 += w * my3sq
            a4 += tl.where(my0 > 0.0, w, 0.0)
            a5 += tl.where(my1 > 0.0, w, 0.0)
            a6 += w * my2
            a7 += w * my3
        if HAS_GRADIENTS:
            # 循环已在 y 处写了 p_y·dx_y，这里覆盖成 (p_y-1)·dx_y。持有列 y 的
            # lane 可能在别的 warp，标量 store 必须等它写完 —— 这条 barrier
            # 不能省（num_warps=1 时几乎免费）。
            tl.debug_barrier()
            g_y = tl.exp(seglu_X_y - m) * inv_d - 1.0
            if reduction == "mean":
                g_y = g_y / n_non_ignore
            c_y = tl.where(my0 > 0.0, 1.0 - t0, 1.0)
            c_y = tl.where(my1 > 0.0, c_y + t1, c_y)
            tl.store(
                X_ptr + y,
                g_y * (c_y - 2.0 * t2 * my2 + 2.0 * t3 * my3),
            )

        if HAS_MULTIMAX_GRADIENTS:
            # One atomic_add per row per scalar (8 total). Race-free across
            # programs. 实测代价：把这 8 条换成 8 条普通 store（无争用）只快
            # 0.43 ms / 25.1 ms = 1.7%，所以原注释的「well below kernel
            # runtime」是对的，这里不是杠杆。
            tl.atomic_add(grad_t_ptr + 0, a0)
            tl.atomic_add(grad_t_ptr + 1, a1)
            tl.atomic_add(grad_t_ptr + 2, a2)
            tl.atomic_add(grad_t_ptr + 3, a3)
            tl.atomic_add(grad_r_ptr + 0, t0 * a4)
            tl.atomic_add(grad_r_ptr + 1, -t1 * a5)
            tl.atomic_add(grad_r_ptr + 2, 2.0 * t2 * a6)
            tl.atomic_add(grad_r_ptr + 3, -2.0 * t3 * a7)

    tl.debug_barrier()

    loss = lse - seglu_X_y

    if reduction == "mean":
        loss = loss / n_non_ignore

    tl.store(loss_ptr, loss)
