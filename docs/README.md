# Documentation index

Start with [the operation and usage workflow](WORKFLOW.md). Repository-wide
environment and preservation rules are in [agents.md](../agents.md).

## Current operating guides

| Document | Read it for |
|---|---|
| [Repository overview](../README.md) | Supported GPUs, quick start and public API |
| [Operation and usage workflow](WORKFLOW.md) | Activate, inspect, run, validate, benchmark and continue development |
| [Code and file map](CODE_MAP.md) | Active code, historical scripts, stable paths and generated artifacts |
| [Environment rules](../agents.md) | Shared-account restrictions, environment setup and implementation limits |
| [RTX backend](RTX6000D.md) | Algorithm, tensor contract, local Python headers and initial-port provenance |
| [Benchmark entry points](../benchmarks/README.md) | Current RTX benchmark versus historical comparison scripts |

## Results and design records

| Document | Scope and status |
|---|---|
| [RTX reduction report](RTX6000D_REDUCE_BENCHMARK.md) | Current optimized reduction policy; paired 13-case measurements from 23 October 2026 |
| [RTX reduction data](RTX6000D_REDUCE_RESULTS.json) | Raw final timing samples, correctness checks, environment and source hashes |
| [RTX low-batch characterization](RTX6000D_LOW_BATCH.md) | 24 full-forward cases, 32 controlled reducer cases, exploratory model and reproduction commands; 24 October 2026 |
| [Low-batch timing samples](RTX6000D_LOW_BATCH_RESULTS.json) | Full-forward/attention/reducer samples and independent FP32 checks |
| [Controlled split samples](RTX6000D_SPLIT_GRID_RESULTS.json) | Row-by-split timings and FP64 weighted-sum checks, including the small S=1 regression |
| [Fixed-split study method](SM120_SPLIT_POLICY_METHOD.md) | Preregistered training/holdout manifest, capacity-only model families, freeze boundary and uncertainty rules |
| [Fixed-split study results](SM120_SPLIT_POLICY_RESULTS.md) | Independent-session outcomes, selected split counts, full-pipeline regret and limitations |
| [Fixed-split study data](data/sm120_split_policy_20261025/) | Reviewed protocol, source identities, frozen predictions and raw measurement/evaluation records |
| [Current DeepInfer applicability](DEEPINFER_APPLICABILITY.md) | 24 October source audit: conditional integration into non-fused split-KV, fused-path limits, contracts and validation plan |
| [DeepInfer source provenance](DEEPINFER_SOURCE_AUDIT.json) | Framework/dependency/main pins, source hashes and compared upstream Git blobs; no upstream runtime measurements |
| [Hopper research readiness](HOPPER_RESEARCH_READINESS.md) | Confirmed precedence finding, synchronization concern, historical evidence audit and pending validation protocol |
| [Initial RTX validation](RTX6000D_VALIDATION.json) | Archived first-port evidence; predates the reduction optimization |
| [H100 TMA/DSM design](TMA_REDUCTION.md) | Preserved H100 experiment, source provenance and unresolved validation work |
| [Original upstream README](UPSTREAM_README.md) | Unedited attribution and historical upstream instructions/results |

The original document filenames and JSON records remain in place so existing
links and source-hash provenance stay usable. Current instructions live in the
workflow; historical snapshots are labeled as such. The upstream README and
legacy comparison scripts are not the setup instructions for this RTX server.

Current timing guidance uses 20 rounds for paired forward measurements and
16 for the standalone reducer grid, completing their timing-order cycles.
Historical 15-round files remain unchanged and retain their documented ordering
limitations. The new fixed-split harness instead requests 15 and rounds each
stage group to complete cycles, producing 21 or 18 effective rounds. See
[the benchmark entry points](../benchmarks/README.md) for the distinction.

New raw runs belong outside the Git checkout, for example
`~/nhl_project/moore_tma_pj/docs/runs/<unique-run-name>/`. Promote only reviewed,
sanitized reports and relevant reproducibility data into this documentation.

## Larger-row and length-distribution study (26 October)

The frozen `constant S=4` policy achieves **1.259x** pooled full-pipeline geometric speedup over the unchanged production heuristic on 12 fresh held-out cases: **1.259x** in B and **1.259x** in C. However, `N01_full` regresses by **12.32% / 12.35%** in B/C. The policy remains offline.

See [the new results](SM120_LENGTH_STUDY_RESULTS.md),
[method](SM120_LENGTH_STUDY_METHOD.md), and
[reproduction steps](WORKFLOW.md#9-reproduce-the-larger-row-and-matched-length-study).
The study preserves the original kernels and does not deploy the fitted policy.
