# Operation steps and usage workflow

This guide describes the current shared four-card RTX 6000D server, migrated
on 27 October 2026. It has no international network access. Run commands in Bash
after connecting to your account. All working files, caches, logs and builds
must stay under `~/nhl_project`; use the existing `nhlpy3` environment. See
[agents.md](../agents.md) for the full restrictions and [the file map](CODE_MAP.md)
for code locations. Credentials do not belong in scripts or documentation. Research milestones
follow [the hardware-neutral objectives](RESEARCH_OBJECTIVES.md); Hopper is an
optional implementation track.

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
Triton 3.5.1. The RTX backend uses Triton's wheel-provided compiler/runtime
components; routine RTX work does not need a system nvcc installation. The current installation already
works; routine runs require no package installation. Activation sets
`PIP_NO_INDEX=1`. Use verified local wheels/sources for any future repair;
never run upstream download/install scripts on this offline shared host.

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
The machine has physical devices 0–3; check them live before selecting one.
Physical GPU `$GPU` becomes logical `cuda:0` in the example and tests.

`check_env.py` reports the interpreter, package versions, import paths, cache
paths, local headers and visible device. It fails if the environment or source
checkout is wrong, or if the first visible GPU is not capability 12.0. It does
not install software or launch a benchmark. Version differences from the
recorded environment need deliberate validation before performance comparisons.

If the editable package is missing, first inspect the environment. When a
package relink is actually needed, run only from this checkout and active venv:

```bash
FLASH_MLA_SKIP_CUDA_BUILD=1 python -m pip install --no-index --no-build-isolation --no-deps -e .
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
python benchmarks/benchmark_sm120_reduce.py --output "$RUN_DIR/timings" --rounds 20
printf 'Results: %s/timings/results.json\n' "$RUN_DIR"
```

The benchmark requires a fresh output directory. The example above creates a
parent run directory for provenance and leaves its `timings/` child for the
benchmark to create. For the low-batch and controlled split research suites,
see [the characterization report](RTX6000D_LOW_BATCH.md). For a new balanced
replication, use the current commands below rather than copying the historical
15-round settings:

```bash
python benchmarks/benchmark_sm120_reduce.py --suite low-batch --check-reference \
  --rounds 20 --output "$RUN_DIR/low_batch"
python benchmarks/benchmark_sm120_split_grid.py \
  --rounds 16 --output "$RUN_DIR/split_grid"
```

The script compares current `sm120.py` with immutable initial RTX commit
`1f168a52a4bce6d34439991ee63070d6a0c49825`, loaded with `git show` into the run
directory. Use a checkout containing that commit; a source ZIP alone is
insufficient. Check availability with:

```bash
git cat-file -e 1f168a52a4bce6d34439991ee63070d6a0c49825:flash_mla_old/sm120.py
```

Both versions receive identical inputs and are checked before timing. The
benchmark records attention, reduction/finish and full-forward timings over
13 shapes. Twenty paired rounds use CUDA events and 128 calls per graph;
this completes the five-function forward and two-function reducer timing-order
cycles. Results include raw samples, medians, ranges, environment and source
hashes. For `benchmark_sm120_split_grid.py`, use `--rounds 16` to complete its
two-choice cycles. The corrected helper rotates order and reverses only after
a complete cycle. Earlier 15-round records used an order rule that could retain
the same first choice in two-function comparisons; preserve those records and
their limitation rather than relabeling them as balanced measurements.
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

## 8. Run the preregistered fixed-split study

The commands below reproduce the published study. Its held-out outcomes are
now public; reusing these workloads after changing a model does not create a
new untouched validation set. A new hypothesis needs a versioned protocol and
new holdout cases fixed before their outcomes are inspected.

Read [the frozen method](SM120_SPLIT_POLICY_METHOD.md) and
[the results report](SM120_SPLIT_POLICY_RESULTS.md) before starting a new study.
The canonical manifest defines 12 training and 8 held-out cases. Each includes
seven fixed split choices, including the observed production default. This
harness uses the unchanged RTX kernels and times preallocated producer, finish
and full pipelines separately. Its requested 15 rounds become 21 for each
seven-choice group, or 18 for a six-choice finish group. The 128 calls within
one graph observation are not independent statistical samples.

### Session A and the model freeze

Activate the existing environment, select a GPU verified idle at that time,
and create a new parent directory. Child output directories must not exist.

```bash
cd ~/nhl_project/moore_tma_pj/kernels/FlashMLA_old_opt
source scripts/rtx6000d_env.sh
nvidia-smi
# Replace 0 with the currently idle physical GPU.
export CUDA_VISIBLE_DEVICES=0
mkdir -p ~/nhl_project/moore_tma_pj/docs/runs
RUN_ROOT=$(mktemp -d ~/nhl_project/moore_tma_pj/docs/runs/split_policy_XXXXXXXX)
printf 'Keep this study directory: %s\n' "$RUN_ROOT"
python benchmarks/benchmark_sm120_split_policy.py \
  --protocol benchmarks/sm120_split_policy_protocol.json \
  --phase train --validate-protocol-only
python benchmarks/benchmark_sm120_split_policy.py \
  --protocol benchmarks/sm120_split_policy_protocol.json \
  --phase train --rounds 15 --repetitions 128 \
  --output "$RUN_ROOT/train_A"
python benchmarks/analyze_sm120_split_policy.py freeze \
  --protocol benchmarks/sm120_split_policy_protocol.json \
  --train "$RUN_ROOT/train_A/results.json" \
  --output "$RUN_ROOT/frozen_policy.json"
FREEZE_SHA=$(sha256sum "$RUN_ROOT/frozen_policy.json" | cut -d ' ' -f 1)
printf '%s\n' "$FREEZE_SHA" > "$RUN_ROOT/frozen_policy.sha256"
```

The analyzer rejects holdout timing rows in training input. It performs
leave-one-whole-case-out selection, writes coefficients and predictions for all
20 cases, and refuses to overwrite the output record. Preserve that record,
its SHA256 and the source hashes before viewing any held-out timings. Changing
features, candidates, thresholds or workloads afterward requires a new study
version and untouched holdout data.

### Independent all-case sessions B and C

Run these at separately acquired idle-GPU opportunities, each in a fresh Python
process. Do not treat two files from one uninterrupted process as independent
sessions. Re-source the environment in a new shell and restore `RUN_ROOT` to
the study directory printed above. Recheck GPU availability for each session;
do not reset a device, kill another job or change clocks. Fixed input seeds are
reused, so this tests timing/session variation rather than new random inputs.

For session B:

```bash
cd ~/nhl_project/moore_tma_pj/kernels/FlashMLA_old_opt
source scripts/rtx6000d_env.sh
: "${RUN_ROOT:?Restore the study directory printed by session A}"
nvidia-smi
# Set this only after checking the current device state.
export CUDA_VISIBLE_DEVICES=0
FREEZE_SHA=$(cat "$RUN_ROOT/frozen_policy.sha256")
python benchmarks/benchmark_sm120_split_policy.py \
  --protocol benchmarks/sm120_split_policy_protocol.json \
  --phase all --rounds 15 --repetitions 128 \
  --freeze-record "$RUN_ROOT/frozen_policy.json" \
  --freeze-record-sha256 "$FREEZE_SHA" \
  --output "$RUN_ROOT/all_B"
```

At the next independent idle-GPU opportunity, run session C:

```bash
cd ~/nhl_project/moore_tma_pj/kernels/FlashMLA_old_opt
source scripts/rtx6000d_env.sh
: "${RUN_ROOT:?Restore the study directory printed by session A}"
nvidia-smi
export CUDA_VISIBLE_DEVICES=0
FREEZE_SHA=$(cat "$RUN_ROOT/frozen_policy.sha256")
python benchmarks/benchmark_sm120_split_policy.py \
  --protocol benchmarks/sm120_split_policy_protocol.json \
  --phase all --rounds 15 --repetitions 128 \
  --freeze-record "$RUN_ROOT/frozen_policy.json" \
  --freeze-record-sha256 "$FREEZE_SHA" \
  --output "$RUN_ROOT/all_C"
```

The harness requires the frozen policy and hash for all/holdout phases. It
checks public API equivalence, independent numerical references and graph
replay for each candidate, and records source identities and GPU process
observations. These observations do not reserve the GPU or detect every kind
of interference. Preserve failed or contention-affected records; defer and
repeat the entire affected session in a new directory rather than removing
unfavorable cases. Neither B nor C may update the fitted policy.

### Evaluate the frozen choices

This step uses only the CPU and standard library:

```bash
python benchmarks/analyze_sm120_split_policy.py evaluate \
  --protocol benchmarks/sm120_split_policy_protocol.json \
  --frozen "$RUN_ROOT/frozen_policy.json" \
  --sessions "$RUN_ROOT/all_B/results.json" "$RUN_ROOT/all_C/results.json" \
  --output "$RUN_ROOT/evaluation.json"
```

Evaluation checks completion/correctness, unchanged source and environment
identity, frozen-policy identity and distinct session records. Report both
session results, all regressions, the support fallback, sampled-grid regret
and the cross-session selector comparison. A minimum over measured candidates
is noisy; it is not a globally optimal split count. With two test sessions,
bootstrap intervals remain exploratory. Full pipeline time is measured
directly; do not add separately measured stage medians or claim model serving,
Hopper TMA or DeepInfer acceleration from this RTX experiment.

Working outputs remain outside the checkout under `~/nhl_project`. Reviewed
study snapshots are published under [docs/data/sm120_split_policy_20261025](data/sm120_split_policy_20261025/),
with interpretation in [the results report](SM120_SPLIT_POLICY_RESULTS.md).
No dependency installation or production-policy update is part of this workflow.

## 9. Reproduce the larger-row and matched-length study

Read the [version 2 method](SM120_LENGTH_STUDY_METHOD.md) and
[results](SM120_LENGTH_STUDY_RESULTS.md). Version 2 reuses the unchanged GPU
harness but uses a new manifest and CPU analyzer. Its 24 cases comprise eight
shape families, each with full, quarter and matched-token skew profiles.
Cross-validation holds out the entire family. B/C contain only the 12 held-out
cases; do not use the version 1 `--phase all` command for this study.

These published held-outs are now observed evidence. Reproducing them checks
repeatability; testing a new hypothesis requires a versioned protocol with
new untouched families, fixed before measuring them.

### Training and freeze

Check device availability first. Use one idle GPU and a new output parent;
the commands do not install packages or change the runtime split policy.

```bash
cd ~/nhl_project/moore_tma_pj/kernels/FlashMLA_old_opt
source scripts/rtx6000d_env.sh
python tests/test_sm120_length_analysis.py
python benchmarks/benchmark_sm120_split_policy.py \
  --protocol benchmarks/sm120_length_study_protocol.json \
  --phase train --validate-protocol-only
nvidia-smi
# Select an idle physical GPU after inspecting its current process state.
export CUDA_VISIBLE_DEVICES=0
mkdir -p ~/nhl_project/moore_tma_pj/docs/runs
LENGTH_RUN=$(mktemp -d ~/nhl_project/moore_tma_pj/docs/runs/length_study_XXXXXXXX)
printf 'Keep this study directory: %s\n' "$LENGTH_RUN"
python benchmarks/benchmark_sm120_split_policy.py \
  --protocol benchmarks/sm120_length_study_protocol.json \
  --phase train --rounds 15 --repetitions 128 \
  --output "$LENGTH_RUN/train_A"
python benchmarks/analyze_sm120_length_study.py freeze \
  --protocol benchmarks/sm120_length_study_protocol.json \
  --train "$LENGTH_RUN/train_A/results.json" \
  --previous docs/data/sm120_split_policy_20261025/frozen_policy.json \
  --output "$LENGTH_RUN/frozen_policy.json"
LENGTH_SHA=$(sha256sum "$LENGTH_RUN/frozen_policy.json" | cut -d ' ' -f 1)
printf '%s\n' "$LENGTH_SHA" > "$LENGTH_RUN/frozen_policy.sha256"
```

Preserve the protocol, analysis/dependency hashes and freeze record before
opening held-out timings. The analyzer refuses training records containing
held-out cases; changing a sibling's phase invalidates the family protocol.
Do not tune features, eligibility or local switching thresholds after this point.

### Two fresh held-out acquisitions

Each session is a separate Python process with a fresh idle-device check.
Restore `LENGTH_RUN` if using a new shell. Source the environment in every new
shell. The benchmark checks GPU processes at boundaries; also monitor device
activity while it runs. If another user's compute process appears, stop only
your own benchmark, preserve its incomplete record and repeat the entire
affected session in a new directory when idle. Do not reset or reserve GPUs.

Session B:

```bash
cd ~/nhl_project/moore_tma_pj/kernels/FlashMLA_old_opt
source scripts/rtx6000d_env.sh
: "${LENGTH_RUN:?Restore the study directory from training}"
nvidia-smi
export CUDA_VISIBLE_DEVICES=0
LENGTH_SHA=$(cat "$LENGTH_RUN/frozen_policy.sha256")
python benchmarks/benchmark_sm120_split_policy.py \
  --protocol benchmarks/sm120_length_study_protocol.json \
  --phase holdout --rounds 15 --repetitions 128 \
  --freeze-record "$LENGTH_RUN/frozen_policy.json" \
  --freeze-record-sha256 "$LENGTH_SHA" --output "$LENGTH_RUN/session_B"
```

At the next idle acquisition, session C:

```bash
cd ~/nhl_project/moore_tma_pj/kernels/FlashMLA_old_opt
source scripts/rtx6000d_env.sh
: "${LENGTH_RUN:?Restore the study directory from training}"
nvidia-smi
export CUDA_VISIBLE_DEVICES=0
LENGTH_SHA=$(cat "$LENGTH_RUN/frozen_policy.sha256")
python benchmarks/benchmark_sm120_split_policy.py \
  --protocol benchmarks/sm120_length_study_protocol.json \
  --phase holdout --rounds 15 --repetitions 128 \
  --freeze-record "$LENGTH_RUN/frozen_policy.json" \
  --freeze-record-sha256 "$LENGTH_SHA" --output "$LENGTH_RUN/session_C"
python benchmarks/analyze_sm120_length_study.py evaluate \
  --protocol benchmarks/sm120_length_study_protocol.json \
  --frozen "$LENGTH_RUN/frozen_policy.json" \
  --sessions "$LENGTH_RUN/session_B/results.json" "$LENGTH_RUN/session_C/results.json" \
  --output "$LENGTH_RUN/evaluation.json"
```

For a CPU-only check of the published raw copies, choose a fresh output filename:

```bash
cd ~/nhl_project/moore_tma_pj/kernels/FlashMLA_old_opt
source scripts/rtx6000d_env.sh
mkdir -p ~/nhl_project/moore_tma_pj/docs/runs
LENGTH_CHECK=$(mktemp -d ~/nhl_project/moore_tma_pj/docs/runs/length_check_XXXXXXXX)
python benchmarks/analyze_sm120_length_study.py evaluate \
  --protocol benchmarks/sm120_length_study_protocol.json \
  --frozen docs/data/sm120_length_study_20261026/frozen_policy.json \
  --sessions docs/data/sm120_length_study_20261026/session_B.json \
             docs/data/sm120_length_study_20261026/session_C.json \
  --output "$LENGTH_CHECK/evaluation.json"
```

The output must match the archived evaluation exactly. Report all profiles and
losses, family-cluster uncertainty, cross-session selectors and unchanged-policy
comparators. Source hashes and CUDA Graph tests do not establish H100 execution
or DeepInfer performance. The [data archive](data/sm120_length_study_20261026/)
keeps raw samples, premeasurement hashes, frozen decisions and monitoring records.


## 10. Work offline and preserve migration provenance

Read [the migration record](MIGRATION_20261027.md) for integrity and validation.
The canonical checkout, venv and compatibility links retain the same absolute
paths. Re-source the helper in each shell; this creates fresh project-local
caches and avoids reusing generated state from the previous GPU host.

Git commits and local bundles work offline. For authorized publication, export
the intended commit as a verified Git bundle under `~/nhl_project`, move it to
an authorized connected workstation, verify its base/commit/file hashes, then
perform a normal fast-forward push there. Transfer the verified publication
record back. Do not put tokens into remote URLs or files or repeatedly try
external package/Git downloads from this server.

The 23–26 October timing records retain their original host/GPU identity.
Migration validation checks correctness and offline operation; acquire a fresh
baseline before reporting performance on this machine. H100 comparison work
and the dated upstream port study are optional and have their own compatibility
requirements; they do not block the current RTX research objectives.

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
