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
| [Initial RTX validation](RTX6000D_VALIDATION.json) | Archived first-port evidence; predates the reduction optimization |
| [H100 TMA/DSM design](TMA_REDUCTION.md) | Preserved H100 experiment, source provenance and unresolved validation work |
| [Original upstream README](UPSTREAM_README.md) | Unedited attribution and historical upstream instructions/results |

The original document filenames and JSON records remain in place so existing
links and source-hash provenance stay usable. Current instructions live in the
workflow; historical snapshots are labeled as such. The upstream README and
legacy comparison scripts are not the setup instructions for this RTX server.

New raw runs belong outside the Git checkout, for example
`~/nhl_project/moore_tma_pj/docs/runs/<unique-run-name>/`. Promote only reviewed,
sanitized reports and relevant reproducibility data into this documentation.
