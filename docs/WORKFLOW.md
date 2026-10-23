# Operation steps and usage workflow

This guide describes the existing shared RTX 6000D server. Run commands in Bash
after connecting to your account. All working files, caches, logs and builds
must stay under `~/nhl_project`; use the existing `nhlpy3` environment. See
[agents.md](../agents.md) for the full restrictions and [the file map](CODE_MAP.md)
for code locations. Credentials do not belong in scripts or documentation.

## 1. Open the checkout and activate the environment

```bash
cd ~/nhl_project/moore_tma_pj/kernels/FlashMLA_old_opt
source scripts/rtx6000d_env.sh
git status --short
```

The helper sources `~/nhl_project/nhlpy3/bin/activate`, sets project-local pip,
Triton, CUDA and Torch-extension caches, sets `TMPDIR`, disables Python bytecode
writes, and adds the extracted Python headers to `CPATH` if available. Source it
in every new shell, including a shell launched by an editor. Executing it with
`bash scripts/rtx6000d_env.sh` would not activate the parent shell.

The recorded working environment is Python 3.12.3, PyTorch 2.9.1+cu128 and
Triton 3.5.1. The server's default `nvcc` is CUDA 12.0.140; the RTX backend uses
Triton's compiler/runtime components instead. The current installation already
works; routine runs require no package installation.

## 2. Select an idle GPU and inspect the environment

```bash
nvidia-smi
# Set this to an idle physical device from the current status, not by habit.
GPU=0
export CUDA_VISIBLE_DEVICES="$GPU"

# Run outside the checkout to avoid copied root egg-info shadowing metadata.
cd ~/nhl_project
python moore_tma_pj/kernels/FlashMLA_old_opt/scripts/check_env.py
cd ~/nhl_project/moore_tma_pj/kernels/FlashMLA_old_opt
```

Check both current utilization and memory use. If no device is available,
defer GPU work or coordinate access; do not terminate jobs or change device
settings. Visibility selects a GPU for this process; it does not reserve it.
Physical GPU `$GPU` becomes logical `cuda:0` in the example and tests.

`check_env.py` reports the interpreter, package versions, import paths, cache
paths, local headers and visible device. It fails if the environment or source
checkout is wrong, or if the first visible GPU is not capability 12.0. It does
not install software or launch a benchmark. Version differences from the
recorded environment need deliberate validation before performance comparisons.

If the editable package is missing, first inspect the environment. When a
package relink is actually needed, run only from this checkout and active venv:

```bash
FLASH_MLA_SKIP_CUDA_BUILD=1 python -m pip install --no-build-isolation --no-deps -e .
```

This skips the Hopper extension and dependency resolution. Do not omit the flag
on this server. Do not upgrade shared dependencies as part of a routine run.
Missing local Python headers are covered by [the header procedure](RTX6000D.md#local-environment-and-headers).

## 3. Run a complete inference example

```bash
python examples/mla_inference.py
python examples/mla_inference.py --dtype fp16
```

The example constructs a three-page cache, 129 live tokens, two query tokens,
16 query heads and one KV head. It creates metadata, runs bottom-right causal
attention, then prints the device, dtype and output/LSE shapes. This is a usage
smoke check; the numerical suite in the next step is the correctness check.

The call sequence is:

```python
from flash_mla_old import get_mla_metadata, flash_mla_with_kvcache

metadata, splits = get_mla_metadata(cache_seqlens, seq_q * heads_q // heads_k, heads_k)
out, lse = flash_mla_with_kvcache(
    q, k_cache, block_table, cache_seqlens, 512,
    metadata, splits, causal=True,
)
```

| Tensor | Shape and requirements |
|---|---|
| `q` | `(batch, seq_q, heads_q, 576)`, BF16 or FP16 |
| `k_cache` | `(num_pages, 64, heads_k, 576)`, same dtype/device; first 512 channels are values |
| `block_table` | `(batch, max_pages)`, CUDA int32, contiguous columns; valid live page IDs |
| `cache_seqlens` | `(batch,)`, contiguous CUDA int32, values from 0 through `max_pages * 64` |
| `out` | `(batch, seq_q, heads_q, 512)`, input dtype |
| `lse` | `(batch, heads_q, seq_q)`, FP32 natural logarithm |

Query heads must be divisible by KV heads. Keep every input and metadata tensor
on the same device; obtain new metadata when switching devices/backends. Channel
dimensions must be contiguous. SM120's empty metadata markers are not Hopper
scheduler tensors. The API is inference-only. For CUDA graphs, warm the exact
shape/dtype specialization first; in-place length changes must remain within
allocated capacity and keep valid live page IDs.

## 4. Run correctness checks before timing

```bash
python tests/test_sm120.py
```

The current suite has seven test methods: attention reference comparisons,
graph replay, input/device checks, mocked Hopper dispatch, 88 standalone
reduction comparisons and the direct-output route check. A skipped required GPU
test makes the command fail. Keep output and LSE checks enabled and retain the
existing tolerances. `tests/test_flash_mla.py` is a historical experiment with
its reference checks commented out; it is not a substitute for this suite.

## 5. Benchmark the reduction and full forward

Create a unique output directory so earlier measurements remain intact:

```bash
mkdir -p ~/nhl_project/moore_tma_pj/docs/runs
RUN_DIR=$(mktemp -d ~/nhl_project/moore_tma_pj/docs/runs/reduce_XXXXXXXX)
git rev-parse HEAD > "$RUN_DIR/head.txt"
git diff --stat > "$RUN_DIR/changes.txt"
python benchmarks/benchmark_sm120_reduce.py --output "$RUN_DIR/timings" --rounds 15
printf 'Results: %s/timings/results.json\n' "$RUN_DIR"
```

The benchmark requires a fresh output directory. The example above creates a
parent run directory for provenance and leaves its `timings/` child for the
benchmark to create. For the low-batch and controlled split research suites,
see [the characterization report](RTX6000D_LOW_BATCH.md).

The script compares current `sm120.py` with immutable initial RTX commit
`1f168a52a4bce6d34439991ee63070d6a0c49825`, loaded with `git show` into the run
directory. Use a checkout containing that commit; a source ZIP alone is
insufficient. Check availability with:

```bash
git cat-file -e 1f168a52a4bce6d34439991ee63070d6a0c49825:flash_mla_old/sm120.py
```

Both versions receive identical inputs and are checked before timing. The
benchmark records attention, reduction/finish and full-forward timings over
13 shapes. Fifteen paired rounds use CUDA events and 128 calls per graph;
results include raw samples, medians, ranges, environment and source hashes.
Optional `--tune` and `--tune-copy` run larger configuration sweeps; give each
its own output directory. `tests/test_sm120.py --benchmark` is only a short
single-case timing convenience, not this paired benchmark.

Read the [published comparison](RTX6000D_REDUCE_BENCHMARK.md) alongside
`results.json`. Report reduction and full-forward changes separately. The
measurements use resident warm buffers and captured allocations; they exclude
JIT, metadata creation, host-to-device transfers and Python allocation overhead.
They do not measure complete model serving latency. Shared utilization and
unlocked clocks can change results; record interference and include regressions.

## 6. Continue development and record the result

1. Read `git status` and preserve existing edits, profiles and copied artifacts.
   Record the starting revision. Keep backups and raw runs outside the checkout
   but under `~/nhl_project`.
2. Use [the code map](CODE_MAP.md) to identify the backend. RTX reduction changes
   usually belong in `sm120.py`; H100 source and build behavior must be preserved.
3. For runtime changes, run the correctness suite, then paired measurements on
   an available GPU. For documentation or file moves, check links, Python/shell
   syntax, entry points and affected paths; do not rerun a long benchmark merely
   to validate prose.
4. Record source hashes and actual environment with measured results. Update
   the report and docs index. Keep old JSON evidence intact unless explicitly
   replacing it with a labeled new run.
5. Review `git diff --check` and the exact diff. Stage explicit paths only.
   Do not commit credentials, local editor settings, copied binaries, unrelated
   logs or profiler captures. Publish through a normal fast-forward push when
   authorized; never overwrite concurrent shared-account work.

Hopper still uses the original `sm_90a` extension build. Its build and runtime
validation require an appropriate Hopper environment and hardware; see the
[H100 continuation notes](TMA_REDUCTION.md). An RTX run or mocked dispatcher
test cannot validate H100 kernel correctness or performance.

## 7. Investigate current upstream without changing the working environment

Start with [the DeepInfer applicability report](DEEPINFER_APPLICABILITY.md)
and its [source manifest](DEEPINFER_SOURCE_AUDIT.json). Record the framework
commit, its `vendor/FlashMLA` gitlink and independent FlashMLA main separately.
Trace the selected model-layer route before changing a combine kernel: fused
decode has no split-KV combine, and non-fused decode gates it on index capacity
and batch-invariance mode. Record actual split counts as well as the gate.

Source-only inspection needs no GPU run or package install. Do not run
DeepInfer's dependency installer in shared nhlpy3: the audited requirements
target a different Torch/CUDA stack and the build script reinstalls NCCL.
Current NVIDIA kernels require SM100-family hardware, unavailable on this
RTX-only server. H100 validation and current DeepInfer validation are separate
tracks. Keep source snapshots and any future audit artifacts under
`~/nhl_project`; preserve both working backends and existing evidence.

After documentation changes, validate links, fenced example syntax, JSON
provenance and `git diff --check`. Runtime experiments remain subject to the
existing hardware/environment and correctness rules.

## Common problems

| Symptom | Check or action |
|---|---|
| Wrong Python, missing Torch or wrong module path | Re-source the helper in this shell; check `sys.executable` and module locations. Do not install globally. |
| `Python.h` missing during Triton launcher compilation | Inspect the project-local extracted headers and `CPATH`; follow the scoped header procedure. |
| `nvcc`/Hopper build failure during an RTX install | Confirm `FLASH_MLA_SKIP_CUDA_BUILD=1`, active nhlpy3 and `--no-build-isolation --no-deps`. |
| Copied CPython 3.11 extension fails under Python 3.12 | It is an old build artifact. RTX uses Triton; H100 needs a matching rebuild on a compatible toolchain. Preserve the old file before cleanup. |
| Installed version appears older inside the checkout | Copied root `.egg-info` can shadow distribution metadata. Inspect from `~/nhl_project` and verify module source/hash as well as version. An editable install's version is stamped at installation, not updated by every Git commit. |
| Benchmark cannot load its baseline | Verify the exact baseline Git object exists; obtain the required history without resetting local work. |
| No available GPU, unstable timing or OOM | Inspect current device usage and input size. Defer or choose an available GPU; do not alter other jobs or device settings. |
| Unsupported input/device error | Check capability, tensor contract and dtype. Do not remove architecture checks to force a Hopper binary onto RTX. |
