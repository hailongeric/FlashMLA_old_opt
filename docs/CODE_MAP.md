# Code and file map

Canonical checkout:
`~/nhl_project/moore_tma_pj/kernels/FlashMLA_old_opt`.
The older `~/nhl_project/moore_tma_pj/FlashMLA_old_opt` entry is a compatibility
symlink. Use the canonical path in new commands and configuration.

```text
FlashMLA_old_opt/
  README.md                       overview and quick start
  agents.md                       shared-server and development rules
  flash_mla_old/
    __init__.py                   public Python exports
    flash_mla_interface.py        device dispatch and lazy Hopper import
    sm120.py                      RTX attention, reduction and validation
  csrc/                           preserved H100 CUDA implementation
    cutlass/                      pinned dependency submodule
  setup.py                        default Hopper build; RTX opt-out flag
  scripts/
    rtx6000d_env.sh                activate nhlpy3; scope generated state
    check_env.py                  inspect environment and import paths
  examples/
    mla_inference.py              complete paged-cache inference example
  tests/
    test_sm120.py                 current RTX correctness and graph checks
    test_flash_mla.py             historical experiment, checks disabled
  benchmarks/
    benchmark_sm120_reduce.py     current paired RTX benchmark
    legacy/                       historical cross-library comparison code
  benchmark/                      compatibility launchers for old paths
  docs/                           guides, design notes and published evidence
```

## Execution flow

1. Source `scripts/rtx6000d_env.sh` in the shell that will run Python.
2. Call `get_mla_metadata` on CUDA int32 cache lengths, then call
   `flash_mla_with_kvcache` with tensors on the same device.
3. `flash_mla_interface.py` chooses capability 9.0 or 12.0 from the input tensor.
   Hopper loads its native extension lazily. RTX imports the Triton module.
4. RTX runs `_split_attention`. A short single-split request writes final output
   directly; other requests finish through `_launch_merge`, which chooses
   `_copy_single` or `_merge_splits`.
5. The caller receives output in the input dtype and FP32 natural-log LSE.

See [the RTX design](RTX6000D.md) for the input contract and
[the reduction report](RTX6000D_REDUCE_BENCHMARK.md) for launch-policy details.
The H100 bindings, metadata scheduler and cluster reduction remain in
`csrc/flash_api.cpp`, `csrc/flash_fwd_mla_metadata.cu` and
`csrc/flash_fwd_mla_kernel.h`; see [the H100 notes](TMA_REDUCTION.md).

## Organization and compatibility

The two historical scripts formerly under `benchmark/` now live in
`benchmarks/legacy/`, preserving their contents. Small launchers retain the old
script names and forward to the new paths. These scripts import the separate
`flash_mla` package and FlashInfer; they do not benchmark this repository's
`flash_mla_old` API. Their CSV/PNG paths are relative to the caller's working
directory. See [legacy restrictions](../benchmarks/legacy/README.md).

The package name, CUDA source paths, active RTX benchmark path, tests and
historical evidence filenames are unchanged. The frozen comparison script can
therefore still load its baseline from Git. No kernel or scheduler is moved or
rewritten by this organization change.

## Files that stay local

| Location | Handling |
|---|---|
| `build/`, `dist/`, `*.egg-info/`, copied `*.so` | Generated or migrated build state; ignored by Git. Do not use as current validation evidence. Preserve before any deliberate cleanup. |
| `.vscode/settings.json`, `tests/log`, `tests/*.ncu-rep` | Existing local user material; leave intact and stage only explicitly selected project files. |
| `~/nhl_project/.cache/`, `~/nhl_project/.tmp/` | Project-local caches and temporary compilation files |
| `~/nhl_project/toolchains/` | Locally extracted headers and other scoped toolchain assets |
| `~/nhl_project/moore_tma_pj/docs/runs/` | New logs and raw measurements, outside this Git checkout |
| `~/nhl_project/moore_tma_pj/docs/rtx6000d_adaptation_20261023/` | Initial-port audit, environment records and backups |
| `~/nhl_project/moore_tma_pj/docs/rtx6000d_reduce_20261023/` | Reduction baseline/candidates/final runs and publication audit |

The wider workspace already groups sibling kernels, references, benchmarks,
runtimes, models, tooling and archives. Its overview and move record are at
`~/nhl_project/moore_tma_pj/README.md` and
`~/nhl_project/moore_tma_pj/docs/REORGANIZATION.md`. Those server-only files are
outside this repository; GitHub links cannot resolve to them.
