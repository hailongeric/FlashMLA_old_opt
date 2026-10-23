# Benchmarks

| Entry | Purpose | Status |
|---|---|---|
| [benchmark_sm120_reduce.py](benchmark_sm120_reduce.py) | Paired RTX attention/reduction/full-forward comparison with frozen baseline | Current; uses `flash_mla_old` and nhlpy3 |
| [legacy/bench_flash_mla.py](legacy/bench_flash_mla.py) | Historical Torch, upstream FlashMLA, FlashInfer and Triton comparison | Archived experimental script; separate dependencies/imports |
| [legacy/visualize.py](legacy/visualize.py) | Plot the legacy `all_perf.csv` | Historical plotting utility |

Follow [the usage workflow](../docs/WORKFLOW.md#5-benchmark-the-reduction-and-full-forward)
to select a GPU, run correctness checks and create a unique output directory.
Use the [reduction report](../docs/RTX6000D_REDUCE_BENCHMARK.md) for measurement
boundaries and interpretation. Raw generated results belong under
`~/nhl_project/moore_tma_pj/docs/runs/`, not in these source directories.

The old `benchmark/` Python paths forward to `legacy/` for compatibility.
They do not select the current RTX comparison automatically.
