from functools import lru_cache
from importlib import import_module
from typing import Optional, Tuple

import torch


def _backend_for(tensor: torch.Tensor) -> str:
    if not tensor.is_cuda:
        raise ValueError("FlashMLA requires CUDA tensors")
    capability = torch.cuda.get_device_capability(tensor.device)
    if capability == (9, 0):
        return "hopper"
    if capability == (12, 0):
        return "sm120"
    raise RuntimeError(f"Unsupported CUDA capability {capability}; supported: Hopper 9.0 and SM120 12.0")


@lru_cache(maxsize=1)
def _hopper_extension():
    try:
        return import_module("flash_mla_old_cuda")
    except ImportError as error:
        raise RuntimeError(
            "The Hopper backend requires flash_mla_old_cuda built for this Python/PyTorch environment. "
            "Build with FLASH_MLA_SKIP_CUDA_BUILD=0 on a compatible Hopper toolchain."
        ) from error


def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create backend-specific opaque scheduling tensors on the lengths' device.

    cache_seqlens: contiguous (batch_size,) int32 CUDA tensor.
    num_heads_per_head_k: seq_len_q * num_heads_q // num_heads_k.
    The returned pair must stay on the same device/backend as the forward call.
    Hopper returns its original scheduler and split-prefix tensors. SM120 returns
    empty markers with shapes (0, 8) and (0,); it schedules from live lengths.
    """
    if _backend_for(cache_seqlens) == "sm120":
        from .sm120 import get_mla_metadata as get_metadata
        return get_metadata(cache_seqlens, num_heads_per_head_k, num_heads_k)
    with torch.cuda.device(cache_seqlens.device):
        return _hopper_extension().get_mla_metadata(cache_seqlens, num_heads_per_head_k, num_heads_k)


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Paged MLA inference, dispatched by q's CUDA capability.

    q: (batch, seq_q, heads_q, 576), FP16 or BF16.
    k_cache: (num_pages, 64, heads_k, 576), same dtype/device as q.
    The first 512 channels of k_cache are the values (head_dim_v=512).
    block_table: (batch, max_pages) int32; cache_seqlens: (batch,) int32.
    Live page IDs must index k_cache, and lengths must be in [0, max_pages*64].
    Causality is bottom-right aligned: key_pos <= cache_len-seq_q+query_pos.
    Metadata must come from get_mla_metadata on the same backend/device.
    softmax_scale defaults to 1/sqrt(576). No backward pass is implemented.

    Returns out (batch, seq_q, heads_q, 512) in q's dtype and natural-log
    softmax_lse (batch, heads_q, seq_q) in FP32. On SM120, fully masked rows
    return zero output and -inf LSE.
    """
    backend = _backend_for(q)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if backend == "sm120":
        from .sm120 import flash_mla_with_kvcache as forward
        return forward(q, k_cache, block_table, cache_seqlens, head_dim_v,
                       tile_scheduler_metadata, num_splits, softmax_scale, causal)
    with torch.cuda.device(q.device):
        out, softmax_lse = _hopper_extension().fwd_kvcache_mla(
            q, k_cache, None, head_dim_v, cache_seqlens, block_table,
            softmax_scale, causal, tile_scheduler_metadata, num_splits,
        )
    return out, softmax_lse
