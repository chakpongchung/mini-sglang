"""CuTeDSL attention backend.

This is a decode-only replacement for the FlashInfer batch-decode wrappers used
by ``FlashInferBackend``. Prefill continues to use FlashInfer's prefill wrapper,
since a CuTeDSL prefill kernel is out of scope here.

The backend reuses ``FlashInferBackend``'s metadata + plan/run plumbing — only
the decode wrapper class is swapped, so the rest of the engine (CUDA-graph
capture, padding, KV-cache plumbing) is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

from minisgl.core import Batch

from .fi import FICaptureData, FIMetadata, FlashInferBackend
from .cutedsl_kernels import (
    BatchDecodeWithPagedKVCacheWrapper as CuteDecodeWrapper,
    CUDAGraphBatchDecodeWithPagedKVCacheWrapper as CuteGraphDecodeWrapper,
)

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


class CuteDSLBackend(FlashInferBackend):
    """FlashInfer prefill + CuTeDSL decode."""

    def __init__(self, config: "ModelConfig") -> None:
        # Reuse FI's prefill wrapper unchanged. Replace the decode wrapper with
        # the CuTeDSL one. The parent constructor builds both wrappers; the
        # cleanest path is to call super().__init__ then overwrite the decode
        # wrapper before any plan() runs.
        super().__init__(config)
        self.decode_wrappers = CuteDecodeWrapper(
            self.float_workspace_buffer,
            use_tensor_cores=self.use_tensor_cores,
            kv_layout="NHD",
        )
        # Keep the int_workspace_buffer pointer in sync with how fi.py reuses it.
        self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer

    def prepare_for_capture(self, batch: Batch) -> None:
        bs = batch.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture
        capture = self.capture
        self.graph_wrappers[bs] = CuteGraphDecodeWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self.use_tensor_cores,
            indptr_buffer=capture.cu_seqlens_k[: bs + 1],
            indices_buffer=capture.indices,
            last_page_len_buffer=capture.one_tensor[:bs],
        )
        self.graph_wrappers[bs]._int_workspace_buffer = self.int_workspace_buffer
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)

    def _initialize_metadata_once(self, metadata: FIMetadata) -> None:
        # FI-specific isinstance gating in the parent breaks once we swap in
        # our wrapper, so dispatch on the wrapper class directly here.
        if metadata.initialized:
            return
        metadata.initialized = True
        self.last_event.synchronize()
        if isinstance(metadata.wrapper, (CuteDecodeWrapper, CuteGraphDecodeWrapper)):
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.dtype,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
            )
            self.last_event.record()
        else:
            # Prefill: fall through to the FlashInfer path.
            metadata.initialized = False  # let parent flip the flag
            super()._initialize_metadata_once(metadata)
