"""Numerics parity test for the CuTile attention kernels.

Calls the cuTile launch helpers directly (bypassing the engine wiring) and
compares results against a pure-PyTorch SDPA reference.

Run as: `python tests/attention/test_cutile_backend.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from minisgl.attention.cutile import _launch_cutile_decode, _launch_cutile_prefill
from minisgl.utils import call_if_main


@dataclass
class _Setup:
    seqlens_k: list[int]
    seqlens_q: list[int]
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype


def _build_paged_kv(num_pages: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype):
    return torch.randn(num_pages, num_kv_heads, head_dim, device="cuda", dtype=dtype)


def _ref_attention(
    *,
    q: torch.Tensor,                  # (sum_q, qo_h, hd)
    k_cache: torch.Tensor,            # (num_pages, kv_h, hd)
    v_cache: torch.Tensor,
    indices: torch.Tensor,            # (sum_kv,)
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """Pure-torch SDPA reference, ragged batch."""
    bs = cu_seqlens_q.numel() - 1
    qo_h = q.shape[1]
    kv_h = k_cache.shape[1]
    head_dim = q.shape[2]
    out = torch.empty_like(q)

    for r in range(bs):
        q_s, q_e = int(cu_seqlens_q[r]), int(cu_seqlens_q[r + 1])
        k_s, k_e = int(cu_seqlens_k[r]), int(cu_seqlens_k[r + 1])
        q_len = q_e - q_s
        kv_len = k_e - k_s
        if q_len == 0 or kv_len == 0:
            continue

        q_r = q[q_s:q_e].float()                                  # (q, qo_h, hd)
        page_idx = indices[k_s:k_e].long()
        k_r = k_cache.index_select(0, page_idx).float()           # (kv, kv_h, hd)
        v_r = v_cache.index_select(0, page_idx).float()

        # GQA expand
        repeat = qo_h // kv_h
        k_r = k_r.repeat_interleave(repeat, dim=1)                # (kv, qo_h, hd)
        v_r = v_r.repeat_interleave(repeat, dim=1)

        q_r_h = q_r.transpose(0, 1)                               # (qo_h, q, hd)
        k_r_h = k_r.transpose(0, 1)                               # (qo_h, kv, hd)
        v_r_h = v_r.transpose(0, 1)

        s = torch.matmul(q_r_h, k_r_h.transpose(-2, -1)) * softmax_scale  # (qo_h, q, kv)
        if causal:
            offset = kv_len - q_len
            i_idx = torch.arange(q_len, device=q.device).view(1, q_len, 1)
            j_idx = torch.arange(kv_len, device=q.device).view(1, 1, kv_len)
            mask = j_idx > (offset + i_idx)
            s = s.masked_fill(mask, float("-inf"))
        p = torch.softmax(s, dim=-1)
        o = torch.matmul(p, v_r_h)                                # (qo_h, q, hd)
        out[q_s:q_e] = o.transpose(0, 1).to(out.dtype)

    return out


def _setup_metadata(setup: _Setup):
    bs = len(setup.seqlens_q)
    cu_q = torch.tensor([0] + setup.seqlens_q, device="cuda", dtype=torch.int32).cumsum_(0)
    cu_k = torch.tensor([0] + setup.seqlens_k, device="cuda", dtype=torch.int32).cumsum_(0)
    seq_q = torch.tensor(setup.seqlens_q, device="cuda", dtype=torch.int32)
    seq_k = torch.tensor(setup.seqlens_k, device="cuda", dtype=torch.int32)

    # Each KV slot points at a unique page in a paged cache. Build a non-trivial
    # permutation so the gather code is actually tested (not identity).
    sum_k = sum(setup.seqlens_k)
    num_pages = sum_k + 16  # extra pages so indices isn't a contiguous prefix
    perm = torch.randperm(num_pages, device="cuda", dtype=torch.int64)[:sum_k]
    indices = perm.to(torch.int32)
    return bs, cu_q, cu_k, seq_q, seq_k, indices, num_pages


def _check_decode(setup: _Setup) -> None:
    assert all(x == 1 for x in setup.seqlens_q), "decode: every seqlen_q == 1"
    bs, cu_q, cu_k, seq_q, seq_k, indices, num_pages = _setup_metadata(setup)

    q = torch.randn(bs, setup.num_qo_heads, setup.head_dim, device="cuda", dtype=setup.dtype)
    k_cache = _build_paged_kv(num_pages, setup.num_kv_heads, setup.head_dim, setup.dtype)
    v_cache = _build_paged_kv(num_pages, setup.num_kv_heads, setup.head_dim, setup.dtype)
    softmax_scale = 1.0 / math.sqrt(setup.head_dim)

    ref = _ref_attention(
        q=q, k_cache=k_cache, v_cache=v_cache, indices=indices,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        softmax_scale=softmax_scale, causal=False,  # decode: full kv visible
    )

    out = torch.full_like(q, float("nan"))
    _launch_cutile_decode(
        q=q, k_cache=k_cache, v_cache=v_cache, out=out,
        indices=indices, cu_seqlens_k=cu_k, seq_lens_k=seq_k,
        softmax_scale=softmax_scale,
        num_qo_heads=setup.num_qo_heads, num_kv_heads=setup.num_kv_heads,
        head_dim=setup.head_dim,
    )
    torch.cuda.synchronize()

    diff = (out.float() - ref.float()).abs()
    print(f"[decode] max_abs_err={diff.max().item():.4e}  mean_abs_err={diff.mean().item():.4e}")
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
    print("[decode] PASS")


def _check_prefill(setup: _Setup) -> None:
    bs, cu_q, cu_k, seq_q, seq_k, indices, num_pages = _setup_metadata(setup)
    sum_q = sum(setup.seqlens_q)

    q = torch.randn(sum_q, setup.num_qo_heads, setup.head_dim, device="cuda", dtype=setup.dtype)
    k_cache = _build_paged_kv(num_pages, setup.num_kv_heads, setup.head_dim, setup.dtype)
    v_cache = _build_paged_kv(num_pages, setup.num_kv_heads, setup.head_dim, setup.dtype)
    softmax_scale = 1.0 / math.sqrt(setup.head_dim)
    max_q_len = max(setup.seqlens_q)

    ref = _ref_attention(
        q=q, k_cache=k_cache, v_cache=v_cache, indices=indices,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        softmax_scale=softmax_scale, causal=True,
    )

    out = torch.full_like(q, float("nan"))
    _launch_cutile_prefill(
        q=q, k_cache=k_cache, v_cache=v_cache, out=out,
        indices=indices, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        seq_lens_q=seq_q, seq_lens_k=seq_k, max_q_len=max_q_len,
        softmax_scale=softmax_scale,
        num_qo_heads=setup.num_qo_heads, num_kv_heads=setup.num_kv_heads,
        head_dim=setup.head_dim,
    )
    torch.cuda.synchronize()

    diff = (out.float() - ref.float()).abs()
    print(f"[prefill] max_abs_err={diff.max().item():.4e}  mean_abs_err={diff.mean().item():.4e}")
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
    print("[prefill] PASS")


@call_if_main(__name__)
def main() -> None:
    torch.manual_seed(0)
    base_setup = _Setup(
        seqlens_k=[64, 96],
        seqlens_q=[64, 96],
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=128,
        dtype=torch.bfloat16,
    )
    decode_setup = _Setup(
        seqlens_k=[64, 96],
        seqlens_q=[1, 1],
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=128,
        dtype=torch.bfloat16,
    )

    print("=== decode ===")
    _check_decode(decode_setup)

    print("=== prefill (no cache) ===")
    _check_prefill(base_setup)

    print("=== prefill (with cache prefix) ===")
    cached_setup = _Setup(
        seqlens_k=[80, 128],
        seqlens_q=[16, 32],
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=128,
        dtype=torch.bfloat16,
    )
    _check_prefill(cached_setup)
