"""Parity test for the CuTeDSL paged batch-decode kernel against a PyTorch reference."""

import pytest
import torch

from minisgl.attention.cutedsl_kernels import paged_decode_attn


def _reference(q, k_cache, v_cache, indices, indptr, scale):
    bs, nh, hd = q.shape
    nkv = k_cache.shape[1]
    gqa = nh // nkv
    out = torch.empty_like(q)
    for b in range(bs):
        s, e = indptr[b].item(), indptr[b + 1].item()
        idx = indices[s:e].long()
        K = k_cache[idx].float()
        V = v_cache[idx].float()
        for h in range(nh):
            kh = h // gqa
            qq = q[b, h].float()
            scores = (qq @ K[:, kh].T) * scale
            p = torch.softmax(scores, dim=-1)
            out[b, h] = (p @ V[:, kh]).to(q.dtype)
    return out


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hd", [64, 128])
@pytest.mark.parametrize("config", [
    # bs, nh, nkv, seqlens
    (1, 8, 2, [37]),
    (3, 8, 2, [13, 64, 100]),
    (4, 16, 16, [1, 2, 3, 4]),     # MHA, no GQA
    (4, 32, 4, [128, 64, 32, 200]),
])
def test_decode_parity(dtype, hd, config):
    bs, nh, nkv, seqlens = config
    assert len(seqlens) == bs
    torch.manual_seed(0)
    total_kv = sum(seqlens)
    num_pages = total_kv + 17
    device = "cuda"

    q = torch.randn(bs, nh, hd, dtype=dtype, device=device) * 0.1
    k_cache = torch.randn(num_pages, nkv, hd, dtype=dtype, device=device) * 0.1
    v_cache = torch.randn(num_pages, nkv, hd, dtype=dtype, device=device) * 0.1
    indices = torch.randperm(num_pages, device=device)[:total_kv].to(torch.int32)
    indptr_cpu = torch.tensor([0] + list(torch.cumsum(torch.tensor(seqlens), 0).tolist()),
                              dtype=torch.int32)
    indptr = indptr_cpu.to(device)
    scale = hd ** -0.5

    out = paged_decode_attn(q, k_cache, v_cache, indices, indptr, scale=scale)
    ref = _reference(q, k_cache, v_cache, indices.cpu(), indptr_cpu, scale)

    err = (out.float() - ref.float()).abs().max().item()
    # bf16/fp16 attention with online softmax — loose tolerance
    assert err < 5e-2, f"max err {err} too large"
