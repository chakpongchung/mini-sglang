"""FlashInfer-compatible wrappers backed by the CuTeDSL paged-decode kernel.

These mimic the subset of the FlashInfer API exercised by ``attention/fi.py``:

  * ``BatchDecodeWithPagedKVCacheWrapper``
  * ``CUDAGraphBatchDecodeWithPagedKVCacheWrapper``

For each wrapper we expose ``.plan(...)`` and ``.run(q, paged_kv_cache=(K, V))``
with the same kwargs that fi.py uses. We also expose the few private attrs that
fi.py pokes (``_int_workspace_buffer``, ``_backend``) for compatibility.

Prefill is intentionally **not** covered: this module is decode-only, mirroring
the scope agreed for the cutedsl backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .decode import paged_decode_attn


@dataclass
class _DecodePlan:
    """State captured by ``.plan(...)`` and consumed by ``.run(...)``."""
    indices: torch.Tensor          # int32, on GPU
    indptr_gpu: torch.Tensor       # int32, on GPU (cu_seqlens_k)
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    scale: float


class BatchDecodeWithPagedKVCacheWrapper:
    """Drop-in replacement for ``flashinfer.BatchDecodeWithPagedKVCacheWrapper``.

    Only the constructor + plan + run surface used by ``attention/fi.py`` is
    implemented. The constructor signature is matched but most kwargs (workspace,
    kv_layout, use_tensor_cores, backend) are accepted and ignored — our kernel
    has no equivalent state.
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_tensor_cores: bool = True,
        kv_layout: str = "NHD",
        backend: str = "auto",
        **_: object,
    ) -> None:
        assert kv_layout == "NHD", f"only NHD layout is supported, got {kv_layout}"
        self._workspace = float_workspace_buffer
        self._use_tensor_cores = use_tensor_cores
        self._backend = backend
        # Provide an int_workspace_buffer attribute because fi.py reuses it.
        self._int_workspace_buffer = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=float_workspace_buffer.device
        )
        self._plan: _DecodePlan | None = None

    def plan(
        self,
        indptr: torch.Tensor,            # cu_seqlens_k, on CPU per fi.py
        indices: torch.Tensor,           # ragged page indices, on GPU
        last_page_len: torch.Tensor,     # ones for page_size=1
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pos_encoding_mode: str = "NONE",
        seq_lens: torch.Tensor | None = None,
        data_type: torch.dtype | None = None,
        q_data_type: torch.dtype | None = None,
        kv_data_type: torch.dtype | None = None,
        non_blocking: bool = True,
        sm_scale: float | None = None,
        **_: object,
    ) -> None:
        assert page_size == 1, "cutedsl wrapper currently supports page_size=1 only"
        assert pos_encoding_mode == "NONE", (
            "cutedsl wrapper does not implement positional encoding fusion"
        )
        device = indices.device
        # FI passes indptr on CPU pinned memory; copy to GPU once here.
        if indptr.device != device:
            indptr_gpu = indptr.to(device, non_blocking=non_blocking)
        else:
            indptr_gpu = indptr
        if sm_scale is None:
            sm_scale = head_dim ** -0.5
        self._plan = _DecodePlan(
            indices=indices,
            indptr_gpu=indptr_gpu,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            scale=sm_scale,
        )

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache,
        **_: object,
    ) -> torch.Tensor:
        plan = self._plan
        assert plan is not None, "must call plan() before run()"
        k4d, v4d = paged_kv_cache  # each: [num_pages, page_size=1, num_kv_heads, head_dim]
        assert k4d.shape[1] == 1 and v4d.shape[1] == 1, "only page_size=1 supported"
        k_cache = k4d.squeeze(1)
        v_cache = v4d.squeeze(1)
        return paged_decode_attn(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            indices=plan.indices,
            indptr=plan.indptr_gpu,
            scale=plan.scale,
        )


class CUDAGraphBatchDecodeWithPagedKVCacheWrapper(BatchDecodeWithPagedKVCacheWrapper):
    """Drop-in for the CUDA-graph variant.

    fi.py provides preallocated ``indptr_buffer``/``indices_buffer``/
    ``last_page_len_buffer`` and reuses them across captures. We accept and
    record those buffers but otherwise behave like the regular decode wrapper —
    the actual graph capture happens at the model level, not here.
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_tensor_cores: bool = True,
        indptr_buffer: torch.Tensor | None = None,
        indices_buffer: torch.Tensor | None = None,
        last_page_len_buffer: torch.Tensor | None = None,
        backend: str = "auto",
        **_: object,
    ) -> None:
        super().__init__(
            float_workspace_buffer,
            use_tensor_cores=use_tensor_cores,
            kv_layout=kv_layout,
            backend=backend,
        )
        self._indptr_buffer = indptr_buffer
        self._indices_buffer = indices_buffer
        self._last_page_len_buffer = last_page_len_buffer
