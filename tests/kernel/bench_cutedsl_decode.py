"""Quick perf sanity: CuTeDSL paged-decode wrapper vs. FlashInfer."""

import time

import torch

import flashinfer

from minisgl.attention.cutedsl_kernels import BatchDecodeWithPagedKVCacheWrapper as CuteWrapper


def _bench(fn, iters: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - s) / iters * 1e6  # us


def main() -> None:
    bs, nh, nkv, hd, seqlen = 32, 32, 4, 128, 1024
    dtype = torch.bfloat16
    device = "cuda"
    torch.manual_seed(0)
    seqlens = [seqlen] * bs
    total_kv = sum(seqlens)
    num_pages = total_kv + 17

    q = torch.randn(bs, nh, hd, dtype=dtype, device=device) * 0.1
    k = torch.randn(num_pages, 1, nkv, hd, dtype=dtype, device=device) * 0.1
    v = torch.randn(num_pages, 1, nkv, hd, dtype=dtype, device=device) * 0.1
    indices = torch.randperm(num_pages, device=device)[:total_kv].to(torch.int32)
    indptr_cpu = torch.tensor([0] + list(torch.cumsum(torch.tensor(seqlens), 0).tolist()),
                              dtype=torch.int32).pin_memory()
    last_page_len_cpu = torch.ones(bs, dtype=torch.int32).pin_memory()
    seq_lens_cpu = torch.tensor(seqlens, dtype=torch.int32).pin_memory()
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    fi_w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="fa2",
        use_tensor_cores=(nh // nkv) >= 4,
    )
    fi_w.plan(
        indptr=indptr_cpu, indices=indices, last_page_len=last_page_len_cpu,
        num_qo_heads=nh, num_kv_heads=nkv, head_dim=hd, page_size=1,
        pos_encoding_mode="NONE", seq_lens=seq_lens_cpu,
        data_type=dtype, q_data_type=dtype, kv_data_type=dtype,
        non_blocking=True,
    )

    ct_w = CuteWrapper(workspace, kv_layout="NHD")
    ct_w.plan(
        indptr=indptr_cpu, indices=indices, last_page_len=last_page_len_cpu,
        num_qo_heads=nh, num_kv_heads=nkv, head_dim=hd, page_size=1,
        pos_encoding_mode="NONE", seq_lens=seq_lens_cpu,
        data_type=dtype, q_data_type=dtype, kv_data_type=dtype,
        non_blocking=True,
    )

    paged = (k, v)
    fi_us = _bench(lambda: fi_w.run(q=q, paged_kv_cache=paged))
    ct_us = _bench(lambda: ct_w.run(q=q, paged_kv_cache=paged))
    print(f"shape: bs={bs} nh={nh} nkv={nkv} hd={hd} seqlen={seqlen}")
    print(f"  FlashInfer : {fi_us:8.1f} us")
    print(f"  CuTeDSL    : {ct_us:8.1f} us  ({ct_us/fi_us:.2f}x slower)")


if __name__ == "__main__":
    main()
