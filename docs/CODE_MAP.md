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
    benchmark_sm120_split_policy.py  fixed-split training and held-out measurements
    analyze_sm120_split_policy.py  CPU-only fit, prediction freeze and evaluation
    sm120_split_policy_protocol.json  preregistered workload and split choices
    legacy/                       historical cross-library comparison code
  benchmark/                      compatibility launchers for old paths
  docs/                           guides, design notes and published evidence
    SM120_SPLIT_POLICY_METHOD.md   frozen study method and selection rules
    SM120_SPLIT_POLICY_RESULTS.md  independent-session results and limitations
    data/sm120_split_policy_20261025/  reviewed raw data and frozen-policy archive
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

## Fixed-split research files

The [measurement harness](../benchmarks/benchmark_sm120_split_policy.py) runs
the unchanged RTX kernels with the choices in the
[canonical protocol](../benchmarks/sm120_split_policy_protocol.json). The
[CPU analyzer](../benchmarks/analyze_sm120_split_policy.py) fits only training
data, freezes predictions, and evaluates two later independent sessions.
This is an offline research policy; it does not change production dispatch,
kernel code or the runtime split heuristic.

Read the [preregistered method](SM120_SPLIT_POLICY_METHOD.md) for the freeze
boundary and model rules, the [results report](SM120_SPLIT_POLICY_RESULTS.md)
for interpretation, and the [study archive](data/sm120_split_policy_20261025/)
for reviewed raw data, source identities and frozen predictions. The
[workflow](WORKFLOW.md#8-run-the-preregistered-fixed-split-study) gives the
exact commands and shared-environment constraints. New working runs remain
outside this checkout under `~/nhl_project`.

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

## Larger-row and matched-length research extension

- `benchmarks/sm120_length_study_protocol.json`: eight new shape families and
  three matched length profiles per family; exact seeds and seven split choices.
- `benchmarks/analyze_sm120_length_study.py`: CPU-only family-grouped selection,
  immutable prediction freeze, held-out-only evaluation and profile comparisons;
  imports unchanged v1 model functions from `analyze_sm120_split_policy.py`.
- `tests/test_sm120_length_analysis.py`: CPU leakage, matched-total, support and
  default-tie checks. No Torch or GPU required.
- [Version 2 method](SM120_LENGTH_STUDY_METHOD.md),
  [results](SM120_LENGTH_STUDY_RESULTS.md) and
  [raw data](data/sm120_length_study_20261026/).

Both studies use `benchmark_sm120_split_policy.py`. The v1 protocol/analyzer
and all previous raw records remain unchanged. Use the version-specific
[workflow](WORKFLOW.md#9-reproduce-the-larger-row-and-matched-length-study);
v2 B/C use `--phase holdout`, while v1 B/C used `--phase all`.

## Current host and research scope

The four-card RTX 6000D host uses the same canonical checkout and venv paths.
See [research objectives](RESEARCH_OBJECTIVES.md) and
[migration record](MIGRATION_20261027.md). The preserved H100 sources remain an
optional comparison track; they do not constrain the active research goals.
The activation helper scopes generated files and disables pip index access.
Original benchmark records and frozen model/source artifacts remain unchanged.
