"""Parity test: CuTeDSL decode wrapper vs. FlashInfer decode wrapper.

Calls both wrappers through the same FI-compatible API on identical inputs and
compares outputs. This is the closest thing to a true drop-in check.
"""

import pytest
import torch

flashinfer = pytest.importorskip("flashinfer")

from minisgl.attention.cutedsl_kernels import (
    BatchDecodeWithPagedKVCacheWrapper as CuteWrapper,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hd", [64, 128])
@pytest.mark.parametrize("config", [
    (1, 8, 2, [37]),
    (3, 8, 2, [13, 64, 100]),
    (4, 16, 16, [1, 2, 3, 4]),
    (8, 32, 4, [128, 64, 32, 200, 1, 17, 256, 5]),
])
def test_decode_matches_flashinfer(dtype, hd, config):
    bs, nh, nkv, seqlens = config
    assert len(seqlens) == bs
    torch.manual_seed(0)
    total_kv = sum(seqlens)
    num_pages = total_kv + 17
    device = "cuda"

    q = torch.randn(bs, nh, hd, dtype=dtype, device=device) * 0.1
    # FI expects [num_pages, page_size=1, num_kv_heads, head_dim] in NHD layout.
    k_cache = torch.randn(num_pages, 1, nkv, hd, dtype=dtype, device=device) * 0.1
    v_cache = torch.randn(num_pages, 1, nkv, hd, dtype=dtype, device=device) * 0.1
    indices_gpu = torch.randperm(num_pages, device=device)[:total_kv].to(torch.int32)
    indptr_cpu = torch.tensor([0] + list(torch.cumsum(torch.tensor(seqlens), 0).tolist()),
                              dtype=torch.int32).pin_memory()
    last_page_len_cpu = torch.ones(bs, dtype=torch.int32).pin_memory()
    seq_lens_cpu = torch.tensor(seqlens, dtype=torch.int32).pin_memory()

    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)

    # FlashInfer
    fi_w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="fa2",
        use_tensor_cores=(nh // nkv) >= 4,
    )
    fi_w.plan(
        indptr=indptr_cpu, indices=indices_gpu, last_page_len=last_page_len_cpu,
        num_qo_heads=nh, num_kv_heads=nkv, head_dim=hd, page_size=1,
        pos_encoding_mode="NONE", seq_lens=seq_lens_cpu,
        data_type=dtype, q_data_type=dtype, kv_data_type=dtype,
        non_blocking=True,
    )
    out_fi = fi_w.run(q=q, paged_kv_cache=(k_cache, v_cache))

    # CuTeDSL
    ct_w = CuteWrapper(workspace, kv_layout="NHD")
    ct_w.plan(
        indptr=indptr_cpu, indices=indices_gpu, last_page_len=last_page_len_cpu,
        num_qo_heads=nh, num_kv_heads=nkv, head_dim=hd, page_size=1,
        pos_encoding_mode="NONE", seq_lens=seq_lens_cpu,
        data_type=dtype, q_data_type=dtype, kv_data_type=dtype,
        non_blocking=True,
    )
    out_ct = ct_w.run(q=q, paged_kv_cache=(k_cache, v_cache))

    err = (out_fi.float() - out_ct.float()).abs().max().item()
    # bf16/fp16 with online softmax — both kernels accumulate similarly
    assert err < 5e-2, f"max err {err} too large vs flashinfer"
