"""CuTeDSL paged batch-decode attention kernel.

Drop-in replacement for FlashInfer's BatchDecodeWithPagedKVCacheWrapper
on the decode hot path. Correctness-first; not yet performance-tuned.

Layout (matches the FI 'NHD', page_size=1 path used by attention/fi.py):
  q:        [bs,        num_qo_heads, head_dim]    bf16/fp16
  k_cache:  [num_pages, num_kv_heads, head_dim]    bf16/fp16  (page_size=1, flattened)
  v_cache:  [num_pages, num_kv_heads, head_dim]    bf16/fp16
  indices:  [total_kv_tokens]                      int32
  indptr:   [bs + 1]                               int32
  out:      [bs,        num_qo_heads, head_dim]    bf16/fp16

The kernel uses one CTA per (qo_head, req) and one warp (32 threads) per CTA.
Each thread owns ``head_dim // 32`` consecutive elements of q/k/v/o and runs
an online-softmax loop over the request's tokens; the scalar QK score is
reduced with a warp shuffle.
"""

from __future__ import annotations

from typing import Tuple

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


_THREADS_PER_BLOCK = 32  # one warp


@cute.kernel
def _decode_kernel(
    Q,
    K_cache,
    V_cache,
    indices,
    indptr,
    Out,
    scale: cutlass.Float32,
    GQA: cutlass.Constexpr,
    EPT: cutlass.Constexpr,
):
    tid = cute.arch.thread_idx()[0]
    qhead = cute.arch.block_idx()[0]
    req = cute.arch.block_idx()[1]
    kv_head = qhead // GQA

    # Per-thread register fragments.
    q = cute.make_rmem_tensor(EPT, cutlass.Float32)
    o = cute.make_rmem_tensor(EPT, cutlass.Float32)
    for i in cutlass.range(EPT, unroll_full=True):
        q[i] = Q[req, qhead, tid * EPT + i].to(cutlass.Float32)
        o[i] = cutlass.Float32(0.0)

    m = cutlass.Float32(-3.4e38)
    l = cutlass.Float32(0.0)

    seq_start = indptr[req]
    seq_end = indptr[req + 1]

    t = seq_start
    while t < seq_end:
        page = indices[t]
        sp = cutlass.Float32(0.0)
        for i in cutlass.range(EPT, unroll_full=True):
            kv = K_cache[page, kv_head, tid * EPT + i].to(cutlass.Float32)
            sp = sp + q[i] * kv
        s = cute.arch.warp_reduction_sum(sp) * scale

        m_new = cute.arch.fmax(m, s)
        alpha = cute.math.exp(m - m_new, fastmath=True)
        p = cute.math.exp(s - m_new, fastmath=True)
        l = l * alpha + p

        for i in cutlass.range(EPT, unroll_full=True):
            vv = V_cache[page, kv_head, tid * EPT + i].to(cutlass.Float32)
            o[i] = o[i] * alpha + p * vv
        m = m_new
        t = t + 1

    inv_l = cutlass.Float32(1.0) / l
    for i in cutlass.range(EPT, unroll_full=True):
        Out[req, qhead, tid * EPT + i] = (o[i] * inv_l).to(Out.element_type)


@cute.jit
def _decode_launcher(
    Q, K_cache, V_cache, indices, indptr, Out,
    scale: cutlass.Float32,
    NUM_QO_HEADS: cutlass.Constexpr,
    BS: cutlass.Constexpr,
    GQA: cutlass.Constexpr,
    EPT: cutlass.Constexpr,
):
    _decode_kernel(
        Q, K_cache, V_cache, indices, indptr, Out, scale, GQA, EPT,
    ).launch(
        grid=(NUM_QO_HEADS, BS, 1),
        block=(_THREADS_PER_BLOCK, 1, 1),
    )


# Compiled-kernel cache keyed on the constexpr-bearing shape signature.
_compiled_cache: dict = {}


def _get_compiled(num_qo_heads: int, bs: int, gqa: int, ept: int,
                  q_dt, kv_dt, out_dt, idx_dt):
    key = (num_qo_heads, bs, gqa, ept, q_dt, kv_dt, out_dt, idx_dt)
    fn = _compiled_cache.get(key)
    if fn is not None:
        return fn

    # Build fake tensors to drive compilation; only shapes/dtypes matter here.
    head_dim = ept * _THREADS_PER_BLOCK
    num_kv_heads = num_qo_heads // gqa
    fq = torch.empty((bs, num_qo_heads, head_dim), dtype=q_dt, device='cuda')
    fk = torch.empty((1, num_kv_heads, head_dim), dtype=kv_dt, device='cuda')
    fv = torch.empty((1, num_kv_heads, head_dim), dtype=kv_dt, device='cuda')
    fi = torch.empty((1,), dtype=idx_dt, device='cuda')
    fp = torch.empty((bs + 1,), dtype=idx_dt, device='cuda')
    fo = torch.empty((bs, num_qo_heads, head_dim), dtype=out_dt, device='cuda')

    fn = cute.compile(
        _decode_launcher,
        from_dlpack(fq), from_dlpack(fk), from_dlpack(fv),
        from_dlpack(fi), from_dlpack(fp), from_dlpack(fo),
        cutlass.Float32(1.0), num_qo_heads, bs, gqa, ept,
    )
    _compiled_cache[key] = fn
    return fn


def paged_decode_attn(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    out: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute paged batch-decode attention.

    Args:
        q: (bs, num_qo_heads, head_dim).
        k_cache, v_cache: (num_pages, num_kv_heads, head_dim) — page_size=1 layout.
        indices: (total_kv_tokens,) int32, ragged page indices.
        indptr:  (bs + 1,) int32, cumulative seqlens (on GPU).
        out: optional output buffer with the same shape/dtype as q.
        scale: optional softmax scale; defaults to 1/sqrt(head_dim).
    """
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    assert indices.is_cuda and indptr.is_cuda
    assert q.dim() == 3 and k_cache.dim() == 3 and v_cache.dim() == 3
    bs, num_qo_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    assert num_qo_heads % num_kv_heads == 0, "num_qo_heads must be divisible by num_kv_heads"
    assert head_dim % _THREADS_PER_BLOCK == 0, (
        f"head_dim={head_dim} must be a multiple of {_THREADS_PER_BLOCK}"
    )
    assert indptr.numel() == bs + 1
    assert indices.dtype == indptr.dtype == torch.int32
    gqa = num_qo_heads // num_kv_heads
    ept = head_dim // _THREADS_PER_BLOCK
    if out is None:
        out = torch.empty_like(q)
    if scale is None:
        scale = head_dim ** -0.5

    fn = _get_compiled(
        num_qo_heads, bs, gqa, ept, q.dtype, k_cache.dtype, out.dtype, indices.dtype
    )
    fn(
        from_dlpack(q), from_dlpack(k_cache), from_dlpack(v_cache),
        from_dlpack(indices), from_dlpack(indptr), from_dlpack(out),
        cutlass.Float32(scale),
        num_qo_heads, bs, gqa, ept,
    )
    return out
