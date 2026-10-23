"""SM120 paged MLA inference, independent of the Hopper CUDA extension.

Values are the first 512 elements of each 576-element key. Each CTA processes
16 grouped query rows over one KV split; a second kernel merges the normalized
partial outputs with log-sum-exp weights. No Hopper DSM/WGMMA is used.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _split_attention(
    Q, K, Table, Lengths, Partial, PartialLSE,
    SQ: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr,
    QS0: tl.constexpr, QS1: tl.constexpr, QS2: tl.constexpr,
    KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
    TS0: tl.constexpr, MAX_PAGES: tl.constexpr, NUM_PAGES: tl.constexpr,
    SPLITS: tl.constexpr, SCALE_LOG2: tl.constexpr, CAUSAL: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    tile, split, batch_kv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch, kv_head = batch_kv // HK, batch_kv % HK
    group: tl.constexpr = HQ // HK
    rows = tile * BM + tl.arange(0, BM)
    qpos, qhead = rows // group, kv_head * group + rows % group
    valid_row = qpos < SQ
    d = tl.arange(0, 512)
    dr = tl.arange(0, 64)
    qbase = batch * QS0 + qpos * QS1 + qhead * QS2
    q = tl.load(Q + qbase[:, None] + d[None, :], valid_row[:, None], 0)
    qr = tl.load(Q + qbase[:, None] + 512 + dr[None, :], valid_row[:, None], 0)
    length = tl.load(Lengths + batch)
    # Defensive bounds for pointer arithmetic. Callers must still supply valid
    # lengths and page IDs; malformed cache contents are not supported inputs.
    length = tl.minimum(tl.maximum(length, 0), MAX_PAGES * 64)
    split_size = tl.cdiv(tl.cdiv(length, BN), SPLITS) * BN
    start = split * split_size
    end = tl.minimum(start + split_size, length)
    m = tl.full((BM,), -float("inf"), tl.float32)
    z = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, 512), tl.float32)
    for offset in range(start, end, BN):
        n = offset + tl.arange(0, BN)
        token_ok = n < end
        page = tl.load(Table + batch * TS0 + n // 64, token_ok, 0)
        token_ok = token_ok & (page >= 0) & (page < NUM_PAGES)
        kbase = page * KS0 + (n % 64) * KS1 + kv_head * KS2
        k = tl.load(K + kbase[None, :] + d[:, None], token_ok[None, :], 0)
        kr = tl.load(K + kbase[None, :] + 512 + dr[:, None], token_ok[None, :], 0)
        scores = (tl.dot(q, k) + tl.dot(qr, kr)) * SCALE_LOG2
        visible = valid_row[:, None] & token_ok[None, :]
        if CAUSAL:
            visible = visible & (n[None, :] <= length - SQ + qpos[:, None])
        scores = tl.where(visible, scores, -float("inf"))
        next_m = tl.maximum(m, tl.max(scores, 1))
        # Empty rows use a finite origin, avoiding -inf - -inf.
        origin = tl.where(next_m == -float("inf"), 0.0, next_m)
        alpha = tl.exp2(m - origin)
        p = tl.exp2(scores - origin[:, None])
        z = z * alpha + tl.sum(p, 1)
        acc *= alpha[:, None]
        acc = tl.dot(p.to(q.dtype), tl.trans(k), acc)
        m = next_m
    denom = tl.where(z > 0, z, 1.0)
    acc /= denom[:, None]
    lse = tl.where(z > 0, m + tl.log2(denom), -float("inf"))
    row_id = (batch * SQ + qpos) * HQ + qhead
    tl.store(Partial + (row_id[:, None] * SPLITS + split) * 512 + d[None, :], acc, valid_row[:, None])
    tl.store(PartialLSE + row_id * SPLITS + split, lse, valid_row)


@triton.jit
def _merge_splits(Partial, PartialLSE, Out, LSE,
                  SQ: tl.constexpr, HQ: tl.constexpr, SPLITS: tl.constexpr,
                  BS: tl.constexpr, BD: tl.constexpr):
    row, dim_tile = tl.program_id(0), tl.program_id(1)
    splits = tl.arange(0, BS)
    dims = dim_tile * BD + tl.arange(0, BD)
    lses = tl.load(PartialLSE + row * SPLITS + splits, splits < SPLITS, -float("inf"))
    maximum = tl.max(lses, 0)
    origin = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp2(lses - origin)
    total = tl.sum(weights, 0)
    denominator = tl.where(total > 0, total, 1.0)
    values = tl.load(Partial + (row * SPLITS + splits[:, None]) * 512 + dims[None, :], splits[:, None] < SPLITS, 0)
    result = tl.sum(values * weights[:, None], 0) / denominator
    tl.store(Out + row * 512 + dims, result)
    if dim_tile == 0:
        batch, rem = row // (SQ * HQ), row % (SQ * HQ)
        qpos, head = rem // HQ, rem % HQ
        natural_lse = tl.where(total > 0, (maximum + tl.log2(denominator)) * 0.6931471805599453, -float("inf"))
        tl.store(LSE + (batch * HQ + head) * SQ + qpos, natural_lse)


def get_mla_metadata(cache_seqlens, num_heads_per_head_k, num_heads_k):
    if cache_seqlens.ndim != 1 or cache_seqlens.dtype != torch.int32 or not cache_seqlens.is_contiguous():
        raise ValueError("cache_seqlens must be a contiguous 1D CUDA int32 tensor")
    if cache_seqlens.numel() == 0 or num_heads_per_head_k <= 0 or num_heads_k <= 0:
        raise ValueError("batch size and head counts must be positive")
    # Opaque backend markers, not a Hopper scheduler. Read lengths at launch so
    # callers can reuse metadata after in-place growth and during graph replay.
    return (torch.empty((0, 8), dtype=torch.int32, device=cache_seqlens.device),
            torch.empty((0,), dtype=torch.int32, device=cache_seqlens.device))


def flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens, head_dim_v,
                         tile_scheduler_metadata, num_splits, softmax_scale, causal):
    if q.ndim != 4 or k_cache.ndim != 4:
        raise ValueError("q and k_cache must be rank-4 tensors")
    batch, sq, hq, dim = q.shape
    pages, page_size, hk, kd = k_cache.shape
    if min(batch, sq, hq, hk, pages) <= 0 or hq % hk:
        raise ValueError("dimensions must be positive and KV heads must divide query heads")
    if (dim, kd, head_dim_v, page_size) != (576, 576, 512, 64):
        raise ValueError("SM120 supports QK dimension 576, value dimension 512 and page size 64")
    if q.dtype not in (torch.float16, torch.bfloat16) or k_cache.dtype != q.dtype:
        raise ValueError("q and k_cache must have the same FP16 or BF16 dtype")
    for name, tensor in (("k_cache", k_cache), ("block_table", block_table),
                         ("cache_seqlens", cache_seqlens),
                         ("tile_scheduler_metadata", tile_scheduler_metadata), ("num_splits", num_splits)):
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same CUDA device as q")
    if q.stride(-1) != 1 or k_cache.stride(-1) != 1:
        raise ValueError("q and k_cache must have contiguous last dimensions")
    if block_table.ndim != 2 or block_table.shape[0] != batch or block_table.shape[1] == 0 or block_table.stride(-1) != 1 or block_table.dtype != torch.int32:
        raise ValueError("block_table must be (batch, max_pages) int32 with contiguous last dimension")
    if cache_seqlens.shape != (batch,) or cache_seqlens.dtype != torch.int32 or not cache_seqlens.is_contiguous():
        raise ValueError("cache_seqlens must be contiguous int32 with shape (batch,)")
    if (tile_scheduler_metadata.shape != (0, 8) or num_splits.shape != (0,)
            or tile_scheduler_metadata.dtype != torch.int32 or num_splits.dtype != torch.int32):
        raise ValueError("SM120 requires metadata returned by get_mla_metadata on this backend")
    if not math.isfinite(softmax_scale):
        raise ValueError("softmax_scale must be finite")
    row_tiles = triton.cdiv(sq * (hq // hk), 16)
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    splits = min(32, triton.cdiv(block_table.shape[1] * 64, 256),
                 max(1, triton.cdiv(2 * sm_count, batch * hk * row_tiles)))
    partial = torch.empty((batch * sq * hq, splits, 512), device=q.device, dtype=torch.float32)
    partial_lse = torch.empty((batch * sq * hq, splits), device=q.device, dtype=torch.float32)
    out = torch.empty((batch, sq, hq, 512), device=q.device, dtype=q.dtype)
    lse = torch.empty((batch, hq, sq), device=q.device, dtype=torch.float32)
    with torch.cuda.device(q.device):
        _split_attention[(row_tiles, splits, batch * hk)](
            q, k_cache, block_table, cache_seqlens, partial, partial_lse,
            sq, hq, hk, *q.stride()[:3], *k_cache.stride()[:3],
            block_table.stride(0), block_table.shape[1], pages,
            splits, softmax_scale * 1.4426950408889634, causal,
            16, 32, num_warps=4, num_stages=1,
        )
        _merge_splits[(batch * sq * hq, 4)](
            partial, partial_lse, out, lse, sq, hq, splits,
            triton.next_power_of_2(splits), 128, num_warps=4,
        )
    return out, lse
