# Migration to the four-card RTX 6000D host

Completed 27 October 2026. The active project remains at
`~/nhl_project/moore_tma_pj/kernels/FlashMLA_old_opt`, with venv
`~/nhl_project/nhlpy3`. The workspace root remains `~/nhl_project/moore_tma_pj`.
The [research objectives](RESEARCH_OBJECTIVES.md) do not require Hopper.

## Transfer and integrity

Copied `moore_tma_pj`, `nhlpy3` and `toolchains` directly between the machines
over their private LAN using rsync over pinned SSH. No internet download,
conda, package installation, dependency upgrade, system setting or global
configuration change was used. The original host remains intact as rollback.
An existing unrelated destination archive was left untouched.

The snapshot contains **93,831 regular files**, **102,137**
entries and **42,255,351,749 bytes** of regular-file content
(39.35 GiB). Source hashes were checked before and after the
copy; destination SHA256, file sizes/modes and symlink targets match exactly.
The complete private manifest has SHA256 `62d723ae0f31061e742f6097edc54131c388f9e459c87d8ccd0e865c91ebe386`.
Private transfer audits remain under `~/nhl_project/migration_20261027_source`
and `~/nhl_project/migration_20261027_destination` on their respective hosts.
Credentials and connection addresses are not published with the repository.

The copy includes Git history and CUTLASS, untracked user files, profiler
captures, research raw data, models/references, local headers and the existing
venv. Source and destination have identical Python interpreter binaries,
Python 3.12.3, matching x86-64 Linux/libc and the same absolute environment
paths. This compatibility was checked for this migration; copying a venv is
not a general substitute for checking ABI, path and shared-library compatibility.
Fresh root-level compilation/runtime caches are used on the new machine.

## New-host validation

- Reused the copied Python 3.12 venv: Torch 2.9.1+cu128, Triton 3.5.1,
  CUDA runtime 12.8; imports resolve to the canonical project checkout.
- Verified four RTX 6000D devices and capability 12.0 / 156 SMs. Validation
  used one initially idle physical GPU, with foreign-process monitoring.
- All seven existing correctness methods passed with zero skips, including
  FP16/BF16 references, empty/masked rows, live-length CUDA Graph replay and
  dispatch checks. H100 dispatch checks are mocks, not H100 execution.
- BF16 and FP16 inference examples passed; four CPU study-analysis tests passed.
- Reanalysis of the archived 26 October records exactly reproduced the stored
  evaluation. This is a CPU reproducibility check, not new-host timing data.
- Git history, CUTLASS revision, compatibility path and unchanged source hashes
  verified; existing user files and the original host were preserved.

The pre-documentation repository revision is
`34a7a82ac811324c2075f538a5c50d9c3cbc6a3e`. H100 CUDA/CUTLASS and SM120 runtime
sources are preserved. The current helper adds the shell-scoped
`PIP_NO_INDEX=1` default; it does not edit pip's global configuration.

## Operation

```bash
cd ~/nhl_project/moore_tma_pj/kernels/FlashMLA_old_opt
source scripts/rtx6000d_env.sh
nvidia-smi
# Choose a currently idle device from 0–3.
export CUDA_VISIBLE_DEVICES=0
python examples/mla_inference.py
python tests/test_sm120.py
```

Use [the full workflow](WORKFLOW.md) for environment inspection, graphs,
benchmarks and offline publication. Keep logs/builds/caches/backups under
`~/nhl_project`; do not change other users' jobs or shared packages.

## Research continuity

The active objectives cover full-pipeline split scheduling, weighted reduction,
length imbalance, graph compatibility and measured integration costs on
available GPUs. Hopper/TMA is an optional implementation/comparison track,
with its original compatibility and correctness requirements preserved.
The audited DeepInfer SM100 port is also optional and does not gate this work.

The 23–26 October results retain their original timestamps, GPU UUIDs, hashes
and numerical/performance claims. They were not rerun as part of migration.
Correctness on the new machine does not transfer old timing numbers to it.
Acquire fresh, uncontended new-host baselines for the next experiment.
