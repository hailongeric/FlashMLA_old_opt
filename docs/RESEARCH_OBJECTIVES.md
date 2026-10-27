# Research objectives

Updated 27 October 2026. The research objective is **reliable split-KV
attention scheduling and weighted reduction for MLA inference**, validated on
the hardware available to the project. Hopper access is not a prerequisite
for this objective, experiment milestones, or the evaluation plan.

## Primary work on the current machine

The current platform is one shared machine with four RTX 6000D GPUs. Continue
using the SM120 backend and existing `~/nhl_project/nhlpy3` venv. Start with
single-GPU experiments, selecting an idle device for each acquisition. Four
installed cards do not establish a distributed implementation or a multi-GPU
speedup; those require a separately defined workload and measurement protocol.

1. Model the combined cost of the partial-attention producer and finish stage.
   Predict useful split counts across small and large row/group counts,
   allocated capacities, live-length distributions and supported head layouts.
2. Compare the production heuristic, simple fixed-split baselines and proposed
   policies on preregistered fresh shape families. Keep matched length profiles
   together during cross-validation and uncertainty estimation. Include every
   regression, support fallback and cross-session selector gap.
3. Improve the weighted reduction or data movement where its measured cost is
   exposed. Evaluate asynchronous movement, locality, layouts or hierarchical
   combination when the selected architecture supports them. Choose mechanisms
   based on correctness and full-pipeline benefit rather than a required GPU
   generation or a required TMA implementation.
4. Preserve FP32 partials, stable rescaling, LSE log-base conventions, empty-row
   semantics, FP16/BF16 accuracy and CUDA Graph replay. Any policy using live
   lengths must state where those features come from and count their cost;
   hidden CPU reads of CUDA lengths are not a free feature.
5. Validate public-call overhead and a representative layer/model boundary
   after operator validation. Add architecture counters and integrated energy
   measurements only with a defined and reproducible measurement method.

The existing 25/26 October measurements were acquired on the previous host.
They remain historical evidence with their original GPU UUIDs and software
records. Migration correctness is not a new performance replication. Establish
a fresh baseline on this four-card host before comparing new optimizations.
The S4 study's approximately 12.3% held-out regression remains an open problem;
the production policy has not been replaced.

## Implementation and optional comparisons

The public API currently supports the separate SM120 and SM90 implementations
described in the [README](../README.md). Removing the Hopper-only research
objective does not change binary compatibility, tensor contracts or dispatch.
Preserve the H100 CUDA/CUTLASS implementation and its history. It is an optional
reference/comparison track; its device-specific execution still requires
compatible hardware and independent correctness validation.

The [dated DeepInfer source audit](DEEPINFER_APPLICABILITY.md) is a conditional
integration study with its own API and architecture requirements. Revalidate
that upstream before a port. Neither that port nor obtaining an H100 is a gate
for the current RTX research milestones. Archived Hopper audits and frozen
experiment methods remain unchanged as provenance.

## Working environment

All server working files, builds, caches, logs, backups and transfer artifacts
must stay under `~/nhl_project`. Conda, global installations, system changes and
dependency upgrades are prohibited. The current server has no international
network access. Reuse the copied venv and local sources; use verified local
artifacts for any future dependency repair. See [agents.md](../agents.md),
[the workflow](WORKFLOW.md), and [migration record](MIGRATION_20261027.md).
