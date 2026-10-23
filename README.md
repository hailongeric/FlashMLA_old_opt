# FlashMLA old kernel cluster reduction experiments

This repository contains an experimental FlashMLA decoding implementation that merges split-KV attention results within CUDA thread-block clusters. It uses TMA bulk copies to exchange partial outputs through distributed shared memory (DSM), followed by a software softmax-weighted reduction. The goal is to reduce global intermediate traffic and the work left for the final combine kernel.

The project was developed on an **NVIDIA H100**. It is an experimental Hopper implementation, not a validated port to the current RTX 6000D machine. The Python package is **`flash_mla_old`**, and the compiled extension is **`flash_mla_old_cuda`**. Other FlashMLA checkouts in the surrounding workspace build different packages, so verify the module actually imported during comparisons.

## Current source changes

The existing kernel work preserved in this revision changes:

- The metadata scheduling overhead from 5 to 3 blocks.
- The host scheduling SM budget to a fixed value of 112.
- The small-workload scheduling and launch budget from 120 to 112, while retaining the existing formulas based on 132.
- The FP32 output and DSM receive layouts to a 520-float physical row stride for 512 logical output values.

These are retained tuning experiments. This documentation update does not establish their correctness or performance on a new GPU. See [the reduction design and continuation notes](docs/TMA_REDUCTION.md) for the algorithm, source map, synchronization concerns, and validation priorities.

## Repository layout

| Path | Purpose |
|---|---|
| `csrc/flash_api.cpp` | Python binding, tensor checks, metadata allocation, scheduling parameters and forward dispatch. |
| `csrc/flash_fwd_mla_metadata.cu` | Split scheduling, cluster eligibility and compressed split counts. |
| `csrc/flash_fwd_mla_kernel.h` | Partial attention, DSM exchange, weighted cluster merge, final combine and launch selection. |
| `csrc/cutlass/` | CUTLASS dependency recorded as a Git submodule. |
| `flash_mla_old/` | Python package interface. |
| `tests/test_flash_mla.py` | Current experimental entry point and reference helpers. |
| `setup.py` | Extension build configuration targeting `compute_90a`/`sm_90a`. |
| `docs/TMA_REDUCTION.md` | Technical design, current limitations and continuation checklist. |
| `docs/UPSTREAM_README.md` | Preserved previous README with upstream context and attribution. |

In the reorganized development workspace, this checkout is located at:

```text
~/nhl_project/moore_tma_pj/kernels/FlashMLA_old_opt
```

The old `moore_tma_pj/FlashMLA_old_opt` path is a compatibility symlink. Repository source paths above remain relative, so a standalone checkout does not need the surrounding workspace layout.

## Existing environment

On the shared server, use the existing environment:

```bash
cd ~/nhl_project/moore_tma_pj/kernels/FlashMLA_old_opt
source ~/nhl_project/nhlpy3/bin/activate
```

Keep operations and generated files under `~/nhl_project`. Do not install conda, change global packages, or follow historical machine-specific installation paths from archived notes. The previous README is retained for provenance; its installation command is not the current shared-server procedure.

At the 23 October 2026 inspection, the current machine reported RTX 6000D GPUs with compute capability 12.0, the existing environment used Python 3.12.3 and did not expose PyTorch, and the default `nvcc` reported CUDA 12.0.140. Copied binary artifacts include CPython 3.11 extensions. These observations do not describe a working build environment for this repository.

The current API explicitly checks for compute capability 9.0, and the build targets Hopper architecture-specific instructions. Do not remove that guard as a substitute for a port. No CUDA compilation, installation, inference, or GPU correctness/performance run was performed while preparing this revision.

## API identity

After a compatible build has been independently established, the package-level API is:

```python
from flash_mla_old import get_mla_metadata, flash_mla_with_kvcache

tile_scheduler_metadata, num_splits = get_mla_metadata(
    cache_seqlens,
    s_q * h_q // h_kv,
    h_kv,
)

out, lse = flash_mla_with_kvcache(
    q,
    kvcache,
    block_table,
    cache_seqlens,
    dv,
    tile_scheduler_metadata,
    num_splits,
    causal=True,
)
```

The implemented kernel configuration uses a 64-token page, QK dimension 576, value dimension 512, and shared K/V storage. BF16 and optional FP16 paths exist, with FP32 accumulation. Some specialization checks use assertions in a build configured with `-DNDEBUG`; the presence of an assertion alone is not evidence of a reliable runtime guard.

## Test and measurement status

The current test script runs one configured case: batch 1, KV length 32768, 128 query heads, one KV head, one query token, causal attention, and fixed sequence lengths. BF16 is the default; the CLI also accepts FP16.

The reference-output calls, output/LSE comparison calls, and timing code are currently commented out. A successful invocation of this script therefore does **not** establish numerical correctness or a benchmark result. Restore and expand those checks in a separate implementation change before relying on new results.

Historical profiler files on the development machine are kept outside this publication's selected source files. There is no newly verified speedup claim for this revision. Results from the sibling `FlashMLA` implementation should not be presented as measurements of this checkout without matching source revisions, imported extensions, shapes and timing boundaries.

## Acknowledgements

This work builds on [DeepSeek FlashMLA](https://github.com/deepseek-ai/FlashMLA), [FlashAttention](https://github.com/Dao-AILab/flash-attention), and [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass). The original repository license is retained in [LICENSE](LICENSE), and the prior README's citation is retained in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).
