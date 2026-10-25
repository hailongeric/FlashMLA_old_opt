# RTX low-batch characterization: 24 October 2026

## Measurement-method update: 25 October 2026

The tables and raw JSON below preserve the original run. A later review found
that the timing helper always ran the baseline first in its two-function
comparisons. The new [confirmation and split-policy report](SM120_SPLIT_POLICY_RESULTS.md)
records an independent same-method repeat and a corrected, balanced-order
confirmation. Use that report for current confirmation evidence; do not treat
the repeated old method as removing its ordering limitation.

This compares the candidate RTX backend with the frozen initial RTX port. These are operator measurements on RTX 6000D, not Hopper/TMA or complete-model results.

## Results

- Full-forward geometric-mean speedup: **1.099x** across 24 cases.
- Retained reducer geometric-mean speedup: **2.085x** across 16 cases; eight direct-output cases eliminate this launch and are excluded.
- Controlled multi-split reducer geometric-mean speedup: **1.694x** across 28 cases.
- 24 of 24 full-forward medians improved, spanning 1.003x to 1.534x. Small improvements are single-session observations, not cross-session confidence claims.
- Both implementations passed the independent FP32 attention reference on all 24 shapes, and FP64 weighted-reduction checks on all 32 synthetic cases.
- Candidate maximum output error against FP32: 0.00403082; maximum natural-log LSE error: 1.91e-06.

## Full-forward sweep

Fixed Sq=1, Hq=128, Hkv=1, QK=576, V=512, 64-token pages, BF16, causal. Values are medians in microseconds. Reducer columns come from a separate paired measurement; they must not be added to full-forward timings or treated as a profiled time fraction.

| Batch | KV length | Splits | Full before | Full after | Speedup | Reduce before | Reduce after |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 1 | 8.234 | 6.616 | 1.245x | 1.077 | eliminated |
| 1 | 256 | 1 | 22.330 | 20.461 | 1.091x | 1.223 | eliminated |
| 1 | 1024 | 4 | 23.914 | 22.780 | 1.050x | 2.518 | 1.356 |
| 1 | 4096 | 16 | 26.734 | 25.677 | 1.041x | 2.947 | 1.971 |
| 1 | 16384 | 32 | 90.281 | 89.191 | 1.012x | 3.779 | 2.630 |
| 1 | 32768 | 32 | 160.175 | 159.151 | 1.006x | 3.722 | 2.712 |
| 2 | 64 | 1 | 8.730 | 6.795 | 1.285x | 1.401 | eliminated |
| 2 | 256 | 1 | 22.584 | 20.522 | 1.100x | 1.403 | eliminated |
| 2 | 1024 | 4 | 25.268 | 23.417 | 1.079x | 3.442 | 1.563 |
| 2 | 4096 | 16 | 50.124 | 48.853 | 1.026x | 3.902 | 2.664 |
| 2 | 16384 | 20 | 167.458 | 166.632 | 1.005x | 3.990 | 2.996 |
| 2 | 32768 | 20 | 321.752 | 320.654 | 1.003x | 3.918 | 2.877 |
| 4 | 64 | 1 | 9.384 | 6.859 | 1.368x | 1.973 | eliminated |
| 4 | 256 | 1 | 23.354 | 20.665 | 1.130x | 1.979 | eliminated |
| 4 | 1024 | 4 | 28.303 | 24.865 | 1.138x | 5.314 | 1.933 |
| 4 | 4096 | 10 | 90.922 | 88.261 | 1.030x | 5.613 | 2.736 |
| 4 | 16384 | 10 | 328.633 | 325.979 | 1.008x | 5.597 | 2.731 |
| 4 | 32768 | 10 | 808.071 | 805.374 | 1.003x | 5.619 | 2.719 |
| 8 | 64 | 1 | 9.794 | 6.383 | 1.534x | 2.732 | eliminated |
| 8 | 256 | 1 | 23.124 | 19.492 | 1.186x | 2.728 | eliminated |
| 8 | 1024 | 4 | 51.536 | 44.788 | 1.151x | 9.096 | 2.347 |
| 8 | 4096 | 5 | 181.289 | 174.836 | 1.037x | 9.444 | 2.681 |
| 8 | 16384 | 5 | 827.311 | 820.808 | 1.008x | 9.443 | 2.721 |
| 8 | 32768 | 5 | 1791.970 | 1787.028 | 1.003x | 9.436 | 2.685 |

Long-KV forward time is dominated by attention in this implementation. For example, B=1, KV=32768 improves only about one microsecond despite a faster reducer. The short-KV direct-output path and small split counts offer more operator-level benefit. This does not establish a universal combine fraction for FlashMLA.

## Controlled split sweep

The split count is forced only for standalone reducer inputs, independent of sequence length and the attention scheduler. FP32 partials and base-2 LSE feed BF16 output. All 32 cases include an empty row; multi-split cases also include masked partials. The S=1 case times the copy path even though short-context forward bypasses it.

| Rows | S=1 | S=2 | S=4 | S=8 | S=9 | S=16 | S=24 | S=32 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.965x | 1.201x | 1.879x | 1.720x | 1.691x | 1.528x | 1.349x | 1.410x |
| 256 | 1.073x | 1.230x | 2.185x | 1.971x | 1.789x | 1.554x | 1.358x | 1.263x |
| 512 | 1.385x | 1.469x | 2.920x | 2.269x | 2.111x | 1.591x | 1.323x | 1.215x |
| 1024 | 1.902x | 1.759x | 3.622x | 2.565x | 2.373x | 1.585x | 1.334x | 1.244x |

Measured regressions are retained:
- Rows=128, S=1: 1.112 -> 1.153 us (3.66% higher latency, 0.041 us).

Standalone S=1 copy measurements describe a different path from short-context direct output. Keep both paths visible when choosing future policies.

## Initial bytes-only model and held-out checks

As a diagnostic, fit `latency_us = a + b * logical_MiB` independently for each backend on S={2,4,16,32}; hold out S={8,9,24} at every row count. S=1 is excluded because it has a different algorithm. Logical bytes count one read of each partial/LSE and one output/LSE write: `4*R*S*(512+1) + R*(2*512+4)`. They are not measured HBM bytes. This exploratory split was selected during analysis and uses 16 training and 12 held-out configurations per backend.

| Backend | a (us) | b (us/MiB) | Held-out mean absolute percentage error | Maximum error |
|---|---:|---:|---:|---:|
| before | 2.583 | 0.197 | 16.8% | 39.2% |
| after | 1.074 | 0.167 | 3.8% | 11.9% |

These errors measure predictive fit on this one experiment, not uncertainty across GPU sessions. The grid includes launch-wave effects, rounded split extents and a dispatch change at S=8/9. A useful next model should account for these terms and validate on new sessions and row counts. This affine diagnostic does not predict Hopper TMA break-even.

## Reproduction and provenance

- Initial RTX baseline commit: `1f168a52a4bce6d34439991ee63070d6a0c49825`.
- Checkout HEAD during measurement: `fe1620954d22cd2f81650cbae073570fcd2e79c4`; benchmark edits were uncommitted and identified by hashes in the raw records.
- Runtime source SHA-256: `764f799a5ac19e9f48c35a1ba64efd7d05fb03d60cede78a47263838dabaedf1`; identical in the two input experiment records.
- GPU: NVIDIA RTX 6000D, 156 SMs; Torch 2.9.1+cu128, Triton 3.5.1, CUDA runtime 12.8.
- UTC intervals: 2026-10-23T16:23:23Z to 2026-10-23T16:24:36Z (forward); 2026-10-23T16:24:39Z to 2026-10-23T16:24:48Z (split grid). The title uses Asia/Shanghai time.
- Each variant uses 128 graph-batched calls and 15 timing rounds for forward, 15 for the split grid, with rotating/reversed order, warm resident buffers and CUDA events. The benchmark does not alter clocks/power settings.
- Timing excludes JIT, metadata preparation and host-to-device transfer. The synthetic reducer grid is cache-resident repeated work; effective logical bandwidth is not device-memory bandwidth.
- Shared-server GPU visibility is not a reservation. Check for competing jobs before each run; the saved results represent one session per suite.

Activate and check an available GPU before running these commands from the repository. Both benchmark programs require a new output directory under `~/nhl_project` and refuse to reuse an existing directory.

```bash
source scripts/rtx6000d_env.sh
nvidia-smi
# Set this to a GPU verified idle immediately before the run.
export CUDA_VISIBLE_DEVICES=0
RUN_ROOT="$HOME/nhl_project/moore_tma_pj/docs/runs/low-batch-$(date +%Y%m%d-%H%M%S)"
python benchmarks/benchmark_sm120_reduce.py --suite low-batch --check-reference \
  --rounds 15 --output "$RUN_ROOT/low_batch"
python benchmarks/benchmark_sm120_split_grid.py --rounds 15 --output "$RUN_ROOT/split_grid"
python benchmarks/summarize_sm120_research.py \
  --low-batch "$RUN_ROOT/low_batch/results.json" \
  --split-grid "$RUN_ROOT/split_grid/results.json" > "$RUN_ROOT/summary.md"
```

The input JSON records contain all raw timing samples and source hashes. The repository documentation index links the published result snapshots and the separate Hopper research-readiness audit.
