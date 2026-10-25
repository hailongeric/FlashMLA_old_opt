# FlashMLA old kernel experiments: H100 and RTX 6000D

This repository retains the experimental H100 TMA/DSM cluster reduction and adds
an independent Triton backend for **RTX 6000D (compute capability 12.0)**. Both use
the `flash_mla_old` Python API. Dispatch follows the input tensor's device.

| GPU | Execution path | Build |
|---|---|---|
| H100 / Hopper, SM90 | Original `flash_mla_old_cuda` extension, including the existing cluster reduction tuning | Default `sm_90a` CUDA build |
| RTX 6000D, SM120 | Triton paged split-KV attention and FP32 weighted merge | Python package only; Triton JIT compiles for the GPU |
| Other architectures | Explicit unsupported-device error | No implicit fallback |

The H100 CUDA sources are preserved byte-for-byte from commit
`10c778c1f1673c344fe57447447653e4a0c623ee`. The RTX path does not execute or rebuild
the copied Hopper binaries. Hopper `compute_90a` code is not forward-compatible
with Blackwell; see [NVIDIA's compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html).

## Start here

- [Operation and usage workflow](docs/WORKFLOW.md): environment checks, inference,
  correctness tests, paired benchmarking and troubleshooting.
- [Documentation index](docs/README.md) and [code map](docs/CODE_MAP.md).
- [Current DeepInfer applicability](docs/DEEPINFER_APPLICABILITY.md): pinned
  upstream audit, eligible combine paths and the SM100 validation requirement.
- [Fixed-split study method](docs/SM120_SPLIT_POLICY_METHOD.md) and
  [results](docs/SM120_SPLIT_POLICY_RESULTS.md): training, frozen policy,
  independent sessions and held-out full-pipeline selection regret.
- [agents.md](agents.md): shared-server environment, limitations and preservation rules.

## Run on this shared server

```bash
cd ~/nhl_project/moore_tma_pj/kernels/FlashMLA_old_opt
source scripts/rtx6000d_env.sh
```

The script activates `~/nhl_project/nhlpy3` and places Python, Triton, CUDA, pip,
and build caches or temporary files under `~/nhl_project`. No conda or global
installation is used. The old `moore_tma_pj/FlashMLA_old_opt` path remains a
compatibility symlink.

The existing environment has Python **3.12.3**, PyTorch **2.9.1+cu128** and
Triton **3.5.1**. Routine use requires no installation. If the editable package
needs relinking, use this checkout and the active environment:

```bash
FLASH_MLA_SKIP_CUDA_BUILD=1 python -m pip install --no-build-isolation --no-deps -e .
```

Use the existing installation when already present. `FLASH_MLA_SKIP_CUDA_BUILD=1`
skips the Hopper extension and CUTLASS submodule command. The RTX path uses the
CUDA runtime and compiler components supplied by the wheels; it does not depend
on the server's default CUDA 12.0 `nvcc`. Copied CPython 3.11 `.so` files are not
usable with Python 3.12 and are not loaded by this path.

After selecting a free physical GPU on the shared machine:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 python examples/mla_inference.py
CUDA_VISIBLE_DEVICES=0 python tests/test_sm120.py
```

Replace `0` with the free GPU index. The example demonstrates the complete API;
the test command runs numerical comparisons, validation, graph replay and dispatch checks.
It returns a nonzero status if tests fail or the required GPU tests are skipped.
See [RTX implementation, validation and limitations](docs/RTX6000D.md).
The [reduction benchmark report](docs/RTX6000D_REDUCE_BENCHMARK.md) compares the
optimized backend against the original RTX revision across 13 shapes, with
separate reduction and full-forward timings.

The [fixed-split study](docs/SM120_SPLIT_POLICY_METHOD.md) compares measured
split choices with the existing scheduler using the unchanged RTX kernels.
Its primary learned features use host-visible dimensions and allocated capacity,
without reading live CUDA lengths back to the CPU. Follow the
[training and evaluation workflow](docs/WORKFLOW.md#8-run-the-preregistered-fixed-split-study)
to keep held-out outcomes separate from model selection. This research does not
install a runtime policy or establish Hopper, DeepInfer or model-level speedups.

## API

```python
from flash_mla_old import get_mla_metadata, flash_mla_with_kvcache

metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)
out, lse = flash_mla_with_kvcache(
    q, kvcache, block_table, cache_seqlens, 512,
    metadata, num_splits, causal=True,
)
```

Inputs use Q/K dimension **576**, value dimension **512**, **64-token pages**,
and FP16 or BF16. Values share the first 512 channels of the key cache. Query
heads must be divisible by KV heads. Output has shape `(batch, seq_q, heads_q,
512)` in the input dtype; LSE has shape `(batch, heads_q, seq_q)` in FP32 and uses
the natural logarithm. Causal masking aligns the query at the end of the cache.
This is an inference-only API.

Metadata is opaque and backend-specific. Always obtain it on the same device as
the forward inputs. SM120 returns empty marker tensors and reads current lengths
at launch; Hopper retains its original scheduler and split-prefix tensors. Do not
transfer metadata between the two backends.

## Preserve and continue H100 work

On an H100 with a compatible PyTorch/CUDA/CUTLASS toolchain, the original build
remains the default:

```bash
FLASH_MLA_SKIP_CUDA_BUILD=0 python -m pip install --no-build-isolation --no-deps -e .
```

The extension is loaded lazily only for Hopper inputs. Existing tuning remains:
metadata overhead 3 blocks; scheduling budget 112 with the historical formulas
based on 132; and a 520-float physical row stride for 512 output elements. The
H100 path was not executed on this RTX-only server. Preserving its code and
testing dispatch does not establish its numerical correctness or speed.

The legacy `tests/test_flash_mla.py` retains its historical experiment, with
reference comparisons and timing commented out. Use the new RTX tests for the
SM120 validation; do not interpret the legacy script alone as a correctness test.

## Source map

| Path | Purpose |
|---|---|
| `flash_mla_old/flash_mla_interface.py` | Public API, capability dispatch and lazy Hopper import |
| `flash_mla_old/sm120.py` | RTX paged attention, stable split merge and input checks |
| `csrc/flash_api.cpp` | Preserved Hopper binding and scheduling parameters |
| `csrc/flash_fwd_mla_metadata.cu` | Preserved Hopper split/cluster scheduling |
| `csrc/flash_fwd_mla_kernel.h` | Preserved Hopper attention, DSM exchange and combine |
| `csrc/cutlass/` | Original CUDA dependency, recorded as a submodule |
| `tests/test_sm120.py` | Attention, reduction edge-case, CUDA graph and dispatch tests |
| `benchmarks/benchmark_sm120_reduce.py` | Paired stage and full-forward timings against the frozen RTX baseline |
| `benchmarks/benchmark_sm120_split_policy.py` | Fixed-split full-pipeline research harness with correctness and frozen-policy checks |
| `benchmarks/analyze_sm120_split_policy.py` | CPU-only training, immutable prediction record and independent-session evaluation |
| `benchmarks/sm120_split_policy_protocol.json` | Fixed 12-training-case and 8-held-out-case manifest |
| `examples/mla_inference.py` | Complete RTX paged-cache inference example |
| `scripts/check_env.py` | Environment, device and import-path preflight |
| `benchmarks/legacy/` | Historical comparison scripts; old paths forward here |
| `scripts/rtx6000d_env.sh` | Shared-server environment and project-local caches |
| `docs/RTX6000D.md` | RTX algorithm, validation and continuation notes |
| `docs/TMA_REDUCTION.md` | Original H100 reduction design and investigation notes |
| `docs/UPSTREAM_README.md` | Preserved upstream context, citations and attribution |

## Acknowledgements

This work builds on [DeepSeek FlashMLA](https://github.com/deepseek-ai/FlashMLA),
[FlashAttention](https://github.com/Dao-AILab/flash-attention),
[NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass), and
[Triton](https://github.com/triton-lang/triton). The original [LICENSE](LICENSE)
and [upstream README](docs/UPSTREAM_README.md) are retained.
