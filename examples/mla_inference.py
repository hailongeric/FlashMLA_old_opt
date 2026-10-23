"""Small paged MLA inference example. Source scripts/rtx6000d_env.sh first."""
import argparse
import json

import torch
from flash_mla_old import get_mla_metadata, flash_mla_with_kvcache


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("Select an available RTX 6000D with CUDA_VISIBLE_DEVICES first.")
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    batch, seq_q, heads_q, heads_k = 1, 2, 16, 1
    live_tokens, pages = 129, 3
    q = torch.randn(batch, seq_q, heads_q, 576, device=device, dtype=dtype)
    k_cache = torch.randn(pages, 64, heads_k, 576, device=device, dtype=dtype)
    block_table = torch.arange(pages, device=device, dtype=torch.int32).reshape(batch, pages)
    cache_seqlens = torch.tensor([live_tokens], device=device, dtype=torch.int32)

    metadata, splits = get_mla_metadata(cache_seqlens, seq_q * heads_q // heads_k, heads_k)
    out, lse = flash_mla_with_kvcache(
        q, k_cache, block_table, cache_seqlens, 512,
        metadata, splits, causal=True,
    )
    assert out.shape == (batch, seq_q, heads_q, 512)
    assert lse.shape == (batch, heads_q, seq_q)
    assert out.dtype == dtype and lse.dtype == torch.float32
    assert torch.isfinite(out).all().item() and torch.isfinite(lse).all().item()
    print(json.dumps({"device": torch.cuda.get_device_name(device), "dtype": args.dtype,
                      "live_tokens": live_tokens, "output_shape": list(out.shape),
                      "lse_shape": list(lse.shape), "finite": True}, indent=2))


if __name__ == "__main__":
    main()
