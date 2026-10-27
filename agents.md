# Agent and contributor instructions

These instructions apply to this repository and work on its shared server.
Read [docs/WORKFLOW.md](docs/WORKFLOW.md) for operation steps and
[docs/CODE_MAP.md](docs/CODE_MAP.md) before reorganizing files. This lowercase
filename is intentional. English Markdown or plain text is the project
documentation format.

## Current research scope

- Follow [the research objectives](docs/RESEARCH_OBJECTIVES.md). The objective
  is split-KV attention scheduling and reduction validated on available GPUs.
  Hopper access or a Hopper TMA implementation is not a project prerequisite.
- The active host is one shared machine with four RTX 6000D GPUs, migrated on
  27 October 2026. Select one idle device per routine test/benchmark. Multi-GPU
  experiments need an explicit protocol; four cards alone imply no speedup.
- Preserve the original H100/CUTLASS work as an optional implementation track.
  Research scope does not waive device dispatch, compatibility or evidence
  requirements. Do not rewrite historical source audits or measured records.

## Authorized environment and setup

- All server working files, edits, downloads, caches, temporary files, build
  output, logs and backups must remain under `/home/ubuntu/nhl_project`
  (`~/nhl_project`). Normal reads of installed system tools and hardware status
  are allowed; do not change system or other users' state.
- Canonical repository:
  `~/nhl_project/moore_tma_pj/kernels/FlashMLA_old_opt`.
  The old `moore_tma_pj/FlashMLA_old_opt` path is a compatibility symlink.
- Reuse `~/nhl_project/nhlpy3`; do not create or install conda. The account and
  this environment are shared, so activating the venv is not permission to
  upgrade its dependencies indiscriminately.
- From the repository, run `source scripts/rtx6000d_env.sh`. This sources
  `~/nhl_project/nhlpy3/bin/activate`, scopes caches/temp paths, disables bytecode
  writes, and configures the existing local Python headers. Source it in each
  execution shell; do not modify shell startup files.
- Migrated venv stack (revalidated on 27 October 2026): Python 3.12.3, PyTorch 2.9.1+cu128, Triton 3.5.1;
  RTX 6000D capability 12.0, 156 SMs, driver 595.99.02. The current machine exposes four cards. Recheck live device state before use.
- The RTX compiler path uses Triton compiler/runtime components from the
  existing wheels. A system nvcc or Hopper toolchain is not needed for it.
  Python headers were extracted under
  `~/nhl_project/toolchains/python3.12-headers/`; no apt installation was used.
- Inspect the live environment with `scripts/check_env.py` from outside the
  checkout as shown in the workflow. Verify imported source locations; copied
  `.egg-info` or CPython 3.11 `.so` files are not reliable current-install proof.
- Only if an editable relink is needed, use the active venv from this checkout:
  `FLASH_MLA_SKIP_CUDA_BUILD=1 python -m pip install --no-index --no-build-isolation --no-deps -e .`.
  Routine tests and benchmarks require no reinstall.

## Offline host and migration

- This host has no international network access. Do not depend on GitHub,
  PyPI, Hugging Face or other external downloads for startup, tests or builds.
  `rtx6000d_env.sh` sets `PIP_NO_INDEX=1` for the active shell. This is a scoped
  offline default, not a global pip configuration change.
- The project, Git objects/submodules, existing venv and local Python headers
  were copied directly over the internal network. No packages were installed
  or upgraded. Reuse `nhlpy3`; do not run an installer to reproduce activation.
- If a dependency repair is needed, use verified local wheels/sources within
  `~/nhl_project`, with pinned hashes and the active venv. Never replace shared
  dependencies or use conda/global tools to work around network restrictions.
- Use an authorized connected workstation for GitHub publication with verified
  bundles/patches. Keep credentials transient and transfer/audit files under
  `~/nhl_project`; do not change `~/.ssh`, global Git settings or startup files.
- Read [the migration record](docs/MIGRATION_20261027.md). The old host remains
  a rollback copy. Old benchmark results are not measurements of the new host.

## Shared-server limitations

- No sudo, global or `--user` pip installations, system apt installations,
  driver/toolkit replacement, global Git/pip configuration, or writes outside
  `~/nhl_project`. Do not use conda. Scope any necessary dependency change to
  nhlpy3 and the authorized task, preserving existing packages where possible.
- Check `nvidia-smi` before GPU work; explicitly select an available device with
  `CUDA_VISIBLE_DEVICES`. Do not kill jobs, reset GPUs, alter clocks/power limits,
  or start multi-GPU workloads merely to accelerate a check. Visibility does
  not reserve a device. Defer work when resources are unavailable.
- Put raw runs and audit backups under
  `~/nhl_project/moore_tma_pj/docs/`, outside this Git checkout. Use unique run
  directories. Never recursively clean shared caches, logs or unknown files.
- Keep passwords, tokens and private connection details out of source, docs,
  commit messages, remote URLs and logs. Use transient authentication when
  publishing is authorized.

## Architecture and preservation rules

- Preserve the user's H100 kernel work and the CUTLASS submodule. The 11 tracked
  CUDA source files under `csrc/` remain identical to preservation commit
  `10c778c1f1673c344fe57447447653e4a0c623ee`. Any intentional H100 change must be
  explicit, separately validated, and documented.
- Keep `flash_mla_old` imports and input-device dispatch stable. Capability 9.0
  selects the lazy Hopper extension; 12.0 selects `sm120.py`. Other devices and
  CPU tensors fail explicitly. Never bypass the guard to run `sm_90a` code on
  RTX. Keep the default Hopper build behavior in `setup.py`.
- The RTX path is separate Triton attention and weighted reduction. It does
  not implement the H100 DSM/TMA cluster exchange. The H100 TMA helper copies
  shared-memory data; software performs the weighted reduction arithmetic.
- Supported RTX inference: FP16/BF16, Q/K dimension 576, V dimension 512,
  64-token pages, shared K/V storage, grouped heads, valid int32 page tables and
  lengths. No backward pass, FP8, separate V cache or arbitrary dimensions.
- Preserve FP32 partials/accumulation and stable LSE weighting. Internal partial
  LSE is base 2; public LSE is natural log. Empty/fully masked RTX rows return
  zero output and `-inf` LSE. Do not mix the log bases or backend metadata.
- Keep CUDA graph replay viable. Forward validation does not synchronize to
  inspect every length/page value; callers must supply valid contents. Avoid
  adding CPU reads or hidden synchronization to the hot path.
- This server has no H100 available for these tests. Source hashes and mocked
  dispatch do not prove H100 runtime correctness or performance. Keep the
  unresolved H100 synchronization questions visible in `docs/TMA_REDUCTION.md`.

## Current upstream investigation

- Consult [the DeepInfer audit](docs/DEEPINFER_APPLICABILITY.md) before assuming
  that this historical dense-MLA API matches current DeepSeek inference.
  Record framework HEAD, its FlashMLA gitlink and separate library HEAD; pin
  source links and hashes. Recheck dispatch when those revisions change.
- Only RTX 6000D is available. Current DeepInfer NVIDIA kernels target SM100
  and the audited FlashMLA APIs require SM100-family hardware. Neither RTX
  results nor future H100 results validate that upstream implementation.
- Do not execute upstream dependency/build installers as part of a source
  audit. The audited DeepInfer installer changes Torch/CUDA dependencies and
  reinstalls NCCL, which would affect shared nhlpy3. Continue source/model
  analysis using the existing environment without package changes.
- Distinguish non-fused split-KV from fused no-split decode. Any future port
  must preserve quantized-cache, attention-sink, ragged-split, empty-row and
  CUDA Graph contracts. Current upstream uses a different empty-LSE sentinel
  from the RTX backend; do not silently replace either backend's semantics.
- A source-level integration opportunity is not a runtime result. Keep
  upstream experiments, historical Hopper work and measured RTX evidence
  explicitly labeled in documentation.

## Fixed-split research protocol

- Read [the preregistered method](docs/SM120_SPLIT_POLICY_METHOD.md) before using
  `benchmarks/benchmark_sm120_split_policy.py`. The canonical manifest is
  `benchmarks/sm120_split_policy_protocol.json`: 12 training cases, 8 held-out
  cases, and seven candidate split counts including production default.
- Complete training session A, fit using only its training rows, and hash the
  frozen policy/prediction record before opening held-out timing outcomes.
  Sessions B and C must be fresh-process all-case runs acquired independently
  on an idle GPU. Reuse the fixed input seeds to isolate session drift.
- Both all-case runs require `--freeze-record` and `--freeze-record-sha256`.
  Preserve the protocol, analyzer, harness, timing helper and method hashes.
  Never refit on B/C, change thresholds using their outcomes, or silently drop
  regressions. A new hypothesis needs a versioned protocol and fresh holdout.
- The split-policy harness requests 15 rounds and rounds each stage group up
  to complete cycles: 21 for seven choices, or 18 for six finish choices when
  the direct-output choice has no finish kernel. This differs from the older
  paired forward and synthetic reducer suites.
- Fit and evaluation are CPU-only. Primary policy features use shape/capacity,
  not host reads of CUDA live lengths. Do not add synchronization or replace
  production dispatch merely because a research policy has been fitted.
- Promote reviewed evidence to [the results report](docs/SM120_SPLIT_POLICY_RESULTS.md)
  and `docs/data/sm120_split_policy_20261025/`. Raw working runs and audit backups
  stay outside this checkout under `~/nhl_project/moore_tma_pj/docs/`.

## Development and validation

1. Inspect `git status`, existing instructions and affected callers before any
   edit. Preserve unrelated user edits, `.vscode/settings.json`, `tests/log`,
   profiler captures and copied build artifacts. Never reset or clean them.
2. Use the stable code layout in `docs/CODE_MAP.md`. Update references when
   moving files. Keep legacy benchmark contents and original evidence records
   distinguishable from current commands/results.
3. Runtime changes: run `python tests/test_sm120.py` on an available RTX GPU;
   require all seven current methods to pass with no required-test skips. Do
   not weaken tolerances. The historical `tests/test_flash_mla.py` has disabled
   reference checks and is insufficient for correctness validation.
4. Performance changes: use `benchmarks/benchmark_sm120_reduce.py --output`
   with a unique directory under `~/nhl_project`, using `--rounds 20` for the
   forward suite and `--rounds 16` for the standalone split grid. These counts
   complete the two- and five-function timing-order cycles. Historical
   15-round records remain evidence with their original ordering limitations.
   Preserve frozen baseline `1f168a52a4bce6d34439991ee63070d6a0c49825` and record
   source hashes, exact shapes/dtypes, environment and all measured regressions.
   Compare reduction and full forward separately; do not claim model or H100
   speedups from RTX operator measurements.
5. Documentation-only changes and path moves: check relative links, syntax and
   affected entry points. Do not spend shared GPU time rerunning an unaffected
   long benchmark. New executable examples should receive a bounded smoke run.
6. Review the exact diff, run `git diff --check`, and stage explicit paths.
   Commit/push only within the user's authorized scope; never force-push over
   concurrent work. Preserve raw evidence and report limitations honestly.

## Larger-row and matched-length protocol (version 2)

- Read [the v2 method](docs/SM120_LENGTH_STUDY_METHOD.md) and
  [results](docs/SM120_LENGTH_STUDY_RESULTS.md). Use
  `sm120_length_study_protocol.json` and `analyze_sm120_length_study.py`;
  preserve the v1 sources and raw archive. The shared GPU harness is unchanged.
- Three matched profiles belong to each shape family. Hold out entire families
  in CV and keep them together in uncertainty estimates; never split sibling
  profiles between training and validation. Quarter/skew totals must match.
- Train A has 12 cases. Freeze before B/C; both B/C use `--phase holdout` for
  the 12 new held-outs, unlike v1's all-case replication. No held-out refit or
  threshold changes. Nonconverged fits are ineligible; a failed final refit
  falls back to default. CPU checks: `python tests/test_sm120_length_analysis.py`.
- All three profiles have the same host-visible shape/capacity, so the offline
  policy must make the same decision for them. No hidden CUDA-length host read
  or runtime deployment is authorized merely by fitting a research model.
- These published cases are now observed. A new model claim requires a new
  protocol and untouched holdouts. Report the old guard, unguarded diagnostic,
  matched-token skew, regressions and full-pipeline results separately.
- Archive reviewed copies in `docs/data/sm120_length_study_20261026/`; all new
  working runs stay outside the checkout under `~/nhl_project/moore_tma_pj/docs/`.
  The existing nhlpy3/shared-server/H100 preservation restrictions still apply.
