"""CuTile attention backend (prototype).

A single-shape feasibility prototype that stands in for the FlashInfer attention
path. Uses NVIDIA CuTile (`cuda.tile`, package `cuda-tile`) to drive paged-KV
prefill and decode kernels. RMSNorm / activation / RoPE / sampling are NOT
covered here -- those continue to use FlashInfer.

Assumptions enforced at construction:
    * `dtype in {torch.float16, torch.bfloat16}`.
    * KV cache `page_size == 1`.
    * `head_dim` is a power of 2 (cuTile tile dims must be powers of 2).
    * `num_qo_heads` is divisible by `num_kv_heads` (GQA).

Out of scope: cuda graph capture, FP8, sliding window, ALiBi, soft-cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


_BLOCK_N_PREFILL = 64
_BLOCK_M_PREFILL = 64
_BLOCK_N_DECODE = 64


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@dataclass
class CuTileMetadata(BaseAttnMetadata):
    cu_seqlens_q_gpu: torch.Tensor   # (bs+1,) int32
    cu_seqlens_k_gpu: torch.Tensor   # (bs+1,) int32
    indices:          torch.Tensor   # (sum_kv_pages,) int32 -- flattened page table
    seq_lens_q_gpu:   torch.Tensor   # (bs,) int32
    seq_lens_k_gpu:   torch.Tensor   # (bs,) int32
    max_q_len:        int
    is_decode:        bool

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class CuTileBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        # Lazy import so users without cuda-tile don't pay an import cost.
        import cuda.tile  # noqa: F401  (raises clear ImportError if missing)

        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device

        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(config.num_qo_heads, tp_size)
        self.num_kv_heads = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        self.head_dim = config.head_dim
        self.softmax_scale = float(config.head_dim) ** -0.5

        if self.kvcache.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(
                f"CuTileBackend only supports fp16/bf16 KV cache, got {self.kvcache.dtype}"
            )
        if not _is_power_of_two(self.head_dim):
            raise NotImplementedError(
                f"CuTileBackend prototype requires head_dim to be a power of 2, got {self.head_dim}"
            )
        if self.num_qo_heads % self.num_kv_heads != 0:
            raise NotImplementedError(
                f"CuTileBackend requires GQA divisibility, got "
                f"num_qo_heads={self.num_qo_heads} num_kv_heads={self.num_kv_heads}"
            )

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        max_q_len = max(seqlens_q)

        device = self.device
        dtype_kw = dict(device=device, dtype=torch.int32)

        seq_lens_q_gpu = torch.tensor(seqlens_q, **dtype_kw)
        seq_lens_k_gpu = torch.tensor(seqlens_k, **dtype_kw)

        cu_seqlens_k_gpu = torch.empty(padded_size + 1, **dtype_kw)
        cu_seqlens_k_gpu[0] = 0
        torch.cumsum(seq_lens_k_gpu, dim=0, out=cu_seqlens_k_gpu[1:].view(-1))

        if max_q_len == 1:
            cu_seqlens_q_gpu = torch.arange(0, padded_size + 1, **dtype_kw)
        elif all(req.cached_len == 0 for req in reqs):
            cu_seqlens_q_gpu = cu_seqlens_k_gpu
        else:
            cu_seqlens_q_gpu = torch.empty(padded_size + 1, **dtype_kw)
            cu_seqlens_q_gpu[0] = 0
            torch.cumsum(seq_lens_q_gpu, dim=0, out=cu_seqlens_q_gpu[1:].view(-1))

        page_table = get_global_ctx().page_table
        indices = torch.cat(
            [page_table[req.table_idx, : req.device_len] for req in reqs]
        ).to(torch.int32)

        batch.attn_metadata = CuTileMetadata(
            cu_seqlens_q_gpu=cu_seqlens_q_gpu,
            cu_seqlens_k_gpu=cu_seqlens_k_gpu,
            indices=indices,
            seq_lens_q_gpu=seq_lens_q_gpu,
            seq_lens_k_gpu=seq_lens_k_gpu,
            max_q_len=max_q_len,
            is_decode=batch.is_decode,
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, CuTileMetadata)

        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        # KV cache view: (num_pages, page_size=1, num_kv_heads, head_dim) -> (num_pages, num_kv_heads, head_dim)
        k_cache = self.kvcache.k_cache(layer_id).squeeze(1)
        v_cache = self.kvcache.v_cache(layer_id).squeeze(1)

        out = torch.empty_like(q)
        if metadata.is_decode:
            _launch_cutile_decode(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                out=out,
                indices=metadata.indices,
                cu_seqlens_k=metadata.cu_seqlens_k_gpu,
                seq_lens_k=metadata.seq_lens_k_gpu,
                softmax_scale=self.softmax_scale,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )
        else:
            _launch_cutile_prefill(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                out=out,
                indices=metadata.indices,
                cu_seqlens_q=metadata.cu_seqlens_q_gpu,
                cu_seqlens_k=metadata.cu_seqlens_k_gpu,
                seq_lens_q=metadata.seq_lens_q_gpu,
                seq_lens_k=metadata.seq_lens_k_gpu,
                max_q_len=metadata.max_q_len,
                softmax_scale=self.softmax_scale,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )
        return out

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        raise NotImplementedError("CuTile backend prototype does not support cuda graphs yet.")

    def prepare_for_capture(self, batch: Batch) -> None:
        raise NotImplementedError("CuTile backend prototype does not support cuda graphs yet.")

    def prepare_for_replay(self, batch: Batch) -> None:
        raise NotImplementedError("CuTile backend prototype does not support cuda graphs yet.")


# -----------------------------------------------------------------------------
# CuTile kernels
# -----------------------------------------------------------------------------

import cuda.tile as ct  # noqa: E402
from cuda.tile import Constant  # noqa: E402

_NEG_INF = float("-inf")


@ct.kernel
def _cutile_decode_kernel(
    q,                # (bs, num_qo_heads, head_dim)
    k_cache,          # (num_pages, num_kv_heads, head_dim)
    v_cache,          # (num_pages, num_kv_heads, head_dim)
    out,              # (bs, num_qo_heads, head_dim)
    page_indices,     # (sum_kv,) int32
    cu_seqlens_k,     # (bs+1,) int32
    seq_lens_k,       # (bs,) int32
    SOFTMAX_SCALE: Constant[float],
    HEAD_DIM:      Constant[int],
    BLOCK_N:       Constant[int],
    KV_HEADS_PER_QO: Constant[int],
):
    req_id  = ct.bid(0)
    qo_head = ct.bid(1)
    kv_head = qo_head // KV_HEADS_PER_QO

    # Load Q row (1, head_dim) for this (request, qo_head)
    q_tile = ct.load(q, (req_id, qo_head, 0), (1, 1, HEAD_DIM))
    q_tile = ct.reshape(q_tile, (1, HEAD_DIM))
    q_tile = ct.astype(q_tile, ct.float32)

    kv_len = ct.load(seq_lens_k, (req_id,), ())
    kv_start = ct.load(cu_seqlens_k, (req_id,), ())

    m_i = ct.full((1,), _NEG_INF, dtype=ct.float32)
    l_i = ct.zeros((1,), dtype=ct.float32)
    o_acc = ct.zeros((1, HEAD_DIM), dtype=ct.float32)

    n_blocks = ct.cdiv(kv_len, BLOCK_N)
    n_iter = ct.arange(BLOCK_N, dtype=ct.int32)
    head_dim_iter = ct.arange(HEAD_DIM, dtype=ct.int32)

    for n in range(n_blocks):
        kv_offs = n * BLOCK_N + n_iter                              # (BLOCK_N,)
        in_range = kv_offs < kv_len                                 # (BLOCK_N,)
        # safe gather: clamp out-of-range to 0; mask their scores
        flat_idx = kv_start + ct.where(
            in_range, kv_offs, ct.zeros((BLOCK_N,), dtype=ct.int32)
        )                                                           # (BLOCK_N,)
        page_idx = ct.gather(page_indices, flat_idx)                # (BLOCK_N,)

        # Build broadcast indices for K/V cache shape (N, kv_h, hd)
        i0 = ct.expand_dims(page_idx, 1)                         # (BLOCK_N, 1)
        i2 = ct.expand_dims(head_dim_iter, 0)                    # (1, HEAD_DIM)
        i1 = ct.full((1, 1), kv_head, dtype=ct.int32)               # broadcasts

        k_tile = ct.gather(k_cache, (i0, i1, i2))                   # (BLOCK_N, HEAD_DIM)
        v_tile = ct.gather(v_cache, (i0, i1, i2))
        k_tile_f32 = ct.astype(k_tile, ct.float32)
        v_tile_f32 = ct.astype(v_tile, ct.float32)

        # S = q @ K^T  : (1, HEAD_DIM) @ (HEAD_DIM, BLOCK_N) -> (1, BLOCK_N)
        k_t = ct.transpose(k_tile_f32, 0, 1)
        s = ct.mma(q_tile, k_t, ct.zeros((1, BLOCK_N), dtype=ct.float32))
        s = s * SOFTMAX_SCALE

        # Mask out-of-range
        in_range_2d = ct.expand_dims(in_range, 0)                # (1, BLOCK_N)
        s = ct.where(in_range_2d, s, ct.full((1, BLOCK_N), _NEG_INF, dtype=ct.float32))

        # Online softmax update
        s_max = ct.max(s, axis=1)                                   # (1,)
        m_new = ct.maximum(m_i, s_max)
        alpha = ct.exp(m_i - m_new)
        p = ct.exp(s - ct.expand_dims(m_new, 1))                 # (1, BLOCK_N)
        l_i = l_i * alpha + ct.sum(p, axis=1)
        o_acc = o_acc * ct.expand_dims(alpha, 1)
        o_acc = ct.mma(p, v_tile_f32, o_acc)
        m_i = m_new

    o_acc = o_acc / ct.expand_dims(l_i, 1)
    o_acc = ct.astype(o_acc, q.dtype)
    o_3d = ct.reshape(o_acc, (1, 1, HEAD_DIM))
    ct.store(out, (req_id, qo_head, 0), o_3d)


@ct.kernel
def _cutile_prefill_kernel(
    q,                # (sum_q, num_qo_heads, head_dim)
    k_cache,          # (num_pages, num_kv_heads, head_dim)
    v_cache,          # (num_pages, num_kv_heads, head_dim)
    out,              # (sum_q, num_qo_heads, head_dim)
    page_indices,     # (sum_kv,) int32
    cu_seqlens_q,     # (bs+1,) int32
    cu_seqlens_k,     # (bs+1,) int32
    seq_lens_q,       # (bs,) int32
    seq_lens_k,       # (bs,) int32
    SOFTMAX_SCALE: Constant[float],
    HEAD_DIM:      Constant[int],
    BLOCK_M:       Constant[int],
    BLOCK_N:       Constant[int],
    KV_HEADS_PER_QO: Constant[int],
):
    req_id  = ct.bid(0)
    qo_head = ct.bid(1)
    m_block = ct.bid(2)
    kv_head = qo_head // KV_HEADS_PER_QO

    q_len  = ct.load(seq_lens_q, (req_id,), ())
    kv_len = ct.load(seq_lens_k, (req_id,), ())
    q_start  = ct.load(cu_seqlens_q, (req_id,), ())
    kv_start = ct.load(cu_seqlens_k, (req_id,), ())

    # If this CTA's M-block is past q_len, do nothing.
    m_offset = m_block * BLOCK_M
    # We can't `return` easily without skipping the store; instead we mask all rows.

    m_iter = ct.arange(BLOCK_M, dtype=ct.int32)        # (BLOCK_M,)
    n_iter = ct.arange(BLOCK_N, dtype=ct.int32)        # (BLOCK_N,)
    head_dim_iter = ct.arange(HEAD_DIM, dtype=ct.int32)

    q_rows = q_start + m_offset + m_iter               # (BLOCK_M,)
    q_in_range = (m_offset + m_iter) < q_len           # (BLOCK_M,)
    q_rows_safe = ct.where(q_in_range, q_rows, ct.zeros((BLOCK_M,), dtype=ct.int32))

    # Q gather: shape (BLOCK_M, HEAD_DIM)
    qi0 = ct.expand_dims(q_rows_safe, 1)            # (BLOCK_M, 1)
    qi2 = ct.expand_dims(head_dim_iter, 0)          # (1, HEAD_DIM)
    qi1 = ct.full((1, 1), qo_head, dtype=ct.int32)
    q_tile = ct.gather(q, (qi0, qi1, qi2))             # (BLOCK_M, HEAD_DIM)
    q_tile = ct.astype(q_tile, ct.float32)

    m_i = ct.full((BLOCK_M,), _NEG_INF, dtype=ct.float32)
    l_i = ct.zeros((BLOCK_M,), dtype=ct.float32)
    o_acc = ct.zeros((BLOCK_M, HEAD_DIM), dtype=ct.float32)

    # Causal-prefix offset: q_pos_abs = (kv_len - q_len) + (m_offset + i)
    q_to_k_offset = kv_len - q_len  # >= 0 for prefix-cached prefill, == 0 for fresh prefill
    abs_q_pos = q_to_k_offset + m_offset + m_iter      # (BLOCK_M,)

    n_blocks = ct.cdiv(kv_len, BLOCK_N)
    for n in range(n_blocks):
        kv_offs = n * BLOCK_N + n_iter                 # (BLOCK_N,)
        in_range_n = kv_offs < kv_len                  # (BLOCK_N,)
        flat_idx = kv_start + ct.where(
            in_range_n, kv_offs, ct.zeros((BLOCK_N,), dtype=ct.int32)
        )
        page_idx = ct.gather(page_indices, flat_idx)   # (BLOCK_N,)

        ki0 = ct.expand_dims(page_idx, 1)           # (BLOCK_N, 1)
        ki2 = ct.expand_dims(head_dim_iter, 0)      # (1, HEAD_DIM)
        ki1 = ct.full((1, 1), kv_head, dtype=ct.int32)
        k_tile = ct.gather(k_cache, (ki0, ki1, ki2))   # (BLOCK_N, HEAD_DIM)
        v_tile = ct.gather(v_cache, (ki0, ki1, ki2))
        k_tile_f32 = ct.astype(k_tile, ct.float32)
        v_tile_f32 = ct.astype(v_tile, ct.float32)

        # S = Q @ K^T : (BLOCK_M, BLOCK_N)
        k_t = ct.transpose(k_tile_f32, 0, 1)
        s = ct.mma(q_tile, k_t, ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32))
        s = s * SOFTMAX_SCALE

        # Mask: valid q row, valid k position, and causal
        i_2d = ct.expand_dims(abs_q_pos, 1)         # (BLOCK_M, 1)
        j_2d = ct.expand_dims(kv_offs, 0)           # (1, BLOCK_N)
        causal_ok = j_2d <= i_2d                       # (BLOCK_M, BLOCK_N)
        n_in_range_2d = ct.expand_dims(in_range_n, 0)   # (1, BLOCK_N)
        m_in_range_2d = ct.expand_dims(q_in_range, 1)   # (BLOCK_M, 1)
        valid = causal_ok & n_in_range_2d & m_in_range_2d
        s = ct.where(valid, s, ct.full((BLOCK_M, BLOCK_N), _NEG_INF, dtype=ct.float32))

        # Online softmax
        s_max = ct.max(s, axis=1)                      # (BLOCK_M,)
        m_new = ct.maximum(m_i, s_max)
        alpha = ct.exp(m_i - m_new)
        p = ct.exp(s - ct.expand_dims(m_new, 1))
        l_i = l_i * alpha + ct.sum(p, axis=1)
        o_acc = o_acc * ct.expand_dims(alpha, 1)
        o_acc = ct.mma(p, v_tile_f32, o_acc)
        m_i = m_new

    # Avoid division by zero for masked-out rows (l_i == 0)
    l_safe = ct.where(l_i > 0, l_i, ct.ones((BLOCK_M,), dtype=ct.float32))
    o_acc = o_acc / ct.expand_dims(l_safe, 1)
    o_acc = ct.astype(o_acc, q.dtype)

    # Scatter back to out, one row per valid m
    oi0 = ct.expand_dims(q_rows_safe, 1)            # (BLOCK_M, 1)
    oi2 = ct.expand_dims(head_dim_iter, 0)          # (1, HEAD_DIM)
    oi1 = ct.full((1, 1), qo_head, dtype=ct.int32)
    # Mask out invalid rows by clamping their indices to 0 -- but this writes to row 0!
    # Correct way: only scatter valid rows. cuTile scatter accepts broadcasted shape;
    # use a sentinel out-of-bounds index for invalid rows so check_bounds drops them.
    invalid_sentinel = ct.full((BLOCK_M,), -1, dtype=ct.int32)
    write_rows = ct.where(q_in_range, q_rows, invalid_sentinel)
    oi0_w = ct.expand_dims(write_rows, 1)
    ct.scatter(out, (oi0_w, oi1, oi2), o_acc)


def _launch_cutile_decode(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
    indices: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seq_lens_k: torch.Tensor,
    softmax_scale: float,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    bs = q.shape[0]
    stream = torch.cuda.current_stream(q.device).cuda_stream
    ct.launch(
        stream,
        (bs, num_qo_heads),
        _cutile_decode_kernel,
        (
            q,
            k_cache,
            v_cache,
            out,
            indices,
            cu_seqlens_k,
            seq_lens_k,
            softmax_scale,
            head_dim,
            _BLOCK_N_DECODE,
            num_qo_heads // num_kv_heads,
        ),
    )


def _launch_cutile_prefill(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
    indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seq_lens_q: torch.Tensor,
    seq_lens_k: torch.Tensor,
    max_q_len: int,
    softmax_scale: float,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    bs = seq_lens_q.shape[0]
    m_blocks = (max_q_len + _BLOCK_M_PREFILL - 1) // _BLOCK_M_PREFILL
    stream = torch.cuda.current_stream(q.device).cuda_stream
    ct.launch(
        stream,
        (bs, num_qo_heads, m_blocks),
        _cutile_prefill_kernel,
        (
            q,
            k_cache,
            v_cache,
            out,
            indices,
            cu_seqlens_q,
            cu_seqlens_k,
            seq_lens_q,
            seq_lens_k,
            softmax_scale,
            head_dim,
            _BLOCK_M_PREFILL,
            _BLOCK_N_PREFILL,
            num_qo_heads // num_kv_heads,
        ),
    )
