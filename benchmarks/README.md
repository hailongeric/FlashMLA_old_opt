# Benchmarks

| Entry | Purpose | Status |
|---|---|---|
| [benchmark_sm120_reduce.py](benchmark_sm120_reduce.py) | Paired RTX attention/reduction/full-forward comparison with frozen baseline | Current; uses `flash_mla_old` and nhlpy3 |
| [benchmark_sm120_split_grid.py](benchmark_sm120_split_grid.py) | Controlled row/split reducer sweep with FP64 reference checks | Current; separate from the attention scheduler |
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
