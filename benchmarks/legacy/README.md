# Historical comparison scripts

These two scripts were moved from `benchmark/` without changing their contents.
They preserve the prior experiment; they are not the supported RTX 6000D
validation or performance entry point.

- `bench_flash_mla.py` imports the separate `flash_mla` package and `flashinfer`,
  and contains an older embedded Triton implementation. It writes
  `all_perf.csv` in the caller's working directory.
- `visualize.py` requires pandas/matplotlib, reads `all_perf.csv` from the caller's
  working directory, and writes `bandwidth_vs_seqlen.png` there.

Do not infer that their dependencies, API contracts or measurements were
validated on this server. Do not install additional libraries just to run them
as part of the current workflow. If deliberately resuming this experiment,
review the script and its reference checks, validate its independent environment,
select an available GPU and use a dedicated working/output directory under
`~/nhl_project` so earlier CSV/PNG files are not overwritten.

The original `benchmark/bench_flash_mla.py` and `benchmark/visualize.py` paths
are compatibility launchers. For current RTX work, use
[the paired reducer benchmark](../benchmark_sm120_reduce.py) and
[the workflow](../../docs/WORKFLOW.md).
