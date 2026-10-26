# Benchmarks

| Entry | Purpose | Status |
|---|---|---|
| [benchmark_sm120_reduce.py](benchmark_sm120_reduce.py) | Paired RTX attention/reduction/full-forward comparison with frozen baseline | Current; uses `flash_mla_old` and nhlpy3 |
| [benchmark_sm120_split_grid.py](benchmark_sm120_split_grid.py) | Controlled row/split reducer sweep with FP64 reference checks | Current; separate from the attention scheduler |
| [benchmark_sm120_split_policy.py](benchmark_sm120_split_policy.py) | Fixed-split producer/finish/full-pipeline measurements with train and frozen holdout phases | Current research harness; unchanged RTX kernels |
| [analyze_sm120_split_policy.py](analyze_sm120_split_policy.py) | Whole-case training, frozen predictions and independent-session evaluation | CPU only; standard library |
| [sm120_split_policy_protocol.json](sm120_split_policy_protocol.json) | Fixed 12-training/8-held-out manifest and seven split choices per case | Preregistered; do not tune using holdout outcomes |
| [summarize_sm120_research.py](summarize_sm120_research.py) | Reproduce the low-batch Markdown tables and exploratory model fit | CPU only; standard library |
| [legacy/bench_flash_mla.py](legacy/bench_flash_mla.py) | Historical Torch, upstream FlashMLA, FlashInfer and Triton comparison | Archived experimental script; separate dependencies/imports |
| [legacy/visualize.py](legacy/visualize.py) | Plot the legacy `all_perf.csv` | Historical plotting utility |

Follow [the usage workflow](../docs/WORKFLOW.md#5-benchmark-the-reduction-and-full-forward)
to select a GPU, run correctness checks and create a unique output directory.
Use the [reduction report](../docs/RTX6000D_REDUCE_BENCHMARK.md) for measurement
boundaries and interpretation. Raw generated results belong under
`~/nhl_project/moore_tma_pj/docs/runs/`, not in these source directories.

The old `benchmark/` Python paths forward to `legacy/` for compatibility.
They do not select the current RTX comparison automatically.

## Research suites

The current paired benchmark keeps its original 13-case default. Select
`--suite low-batch --check-reference` for the 24-shape batch/context sweep
and independent FP32 attention checks. `benchmark_sm120_split_grid.py`
provides 32 synthetic reducer shapes with independent FP64 checks.
Both programs require a fresh `--output` directory under `~/nhl_project`;
an existing directory is rejected to protect earlier evidence.

See [the complete measured report](../docs/RTX6000D_LOW_BATCH.md) for commands,
timing definitions, source hashes, all regressions and model limitations.
These suites measure the RTX backend. The preserved Hopper path needs the
separate [readiness protocol](../docs/HOPPER_RESEARCH_READINESS.md).

## Timing order and split-policy study

For new paired forward runs, pass `--rounds 20`; for the standalone synthetic
reducer grid, pass `--rounds 16`. These complete the five- and two-function
timing-order cycles. Historical 15-round records remain unchanged; their old
rotate/reverse rule could leave the same choice first in two-function groups.
Use the corrected helper and retain the ordering caveat when comparing them.

The fixed-split harness uses separate full, producer and finish groups. Its
default requested count is 15, rounded up to complete cycles: 21 for seven
choices and 18 for six finish choices when the direct path omits a finish.
Each round times 128 graph-replayed calls with resident buffers. Full-pipeline
time is measured directly; separate stage medians are not additive estimates.

Follow the [exact train and evaluation commands](../docs/WORKFLOW.md#8-run-the-preregistered-fixed-split-study):
training session A, CPU-only model fit and prediction freeze, two independent
fresh-process all-case sessions B/C, then CPU-only evaluation. All/holdout
phases require both `--freeze-record` and `--freeze-record-sha256`.
The [preregistered method](../docs/SM120_SPLIT_POLICY_METHOD.md) fixes workload
selection, model families, validation and fallback rules. The
[results report](../docs/SM120_SPLIT_POLICY_RESULTS.md) interprets the reviewed
records in [the study data directory](../docs/data/sm120_split_policy_20261025/).

Use only the existing environment activated by `source scripts/rtx6000d_env.sh`
from the repository root. Select an idle GPU for each measurement session and
use fresh output directories under `~/nhl_project`; do not install packages,
change clocks or alter production dispatch. These measurements concern the RTX
operator, not H100/TMA, current DeepInfer or end-to-end model performance.

## Larger-row and length-distribution study (26 October)

The frozen `constant S=4` policy achieves **1.259x** pooled full-pipeline geometric speedup over the unchanged production heuristic on 12 fresh held-out cases: **1.259x** in B and **1.259x** in C. However, `N01_full` regresses by **12.32% / 12.35%** in B/C. The policy remains offline.

See [the new results](../docs/SM120_LENGTH_STUDY_RESULTS.md),
[method](../docs/SM120_LENGTH_STUDY_METHOD.md), and
[reproduction steps](../docs/WORKFLOW.md#9-reproduce-the-larger-row-and-matched-length-study).
The study preserves the original kernels and does not deploy the fitted policy.
