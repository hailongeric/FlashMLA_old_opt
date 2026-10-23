# Split KV reduction design and continuation notes

> These notes describe the preserved H100 implementation. The separate RTX 6000D backend and its GPU validation are documented in [RTX6000D.md](RTX6000D.md).

## Scope and provenance

These notes describe the `FlashMLA_old_opt` working-tree kernel changes reviewed on 23 October 2026, based on commit `91a10bbffb0cae3e4805d19f07fc1a9632777541`. The earlier history includes `e9db64f` for the cluster-reduction version and `a2d0c78` as a baseline candidate. The project owner confirmed that the original experiments were performed on H100.

The three existing source edits are preserved without rewriting kernel behavior. Documentation and static publication checks are not a new correctness or performance validation. This older kernel family differs from the later `FlashMLA` and FlashAttention experiments in the surrounding workspace.

## Why the merge is weighted

Split-KV decoding allows several CTAs to process different portions of one query's KV sequence. Split `s` produces a normalized output vector `o_s` and log-sum-exp `L_s`. The normalized vectors cannot simply be added because each split uses a different softmax normalization factor.

With natural-log LSE values, the stable merge is:

```text
m   = max_s L_s
a_s = exp(L_s - m)
Z   = sum_s a_s
o   = sum_s(a_s * o_s) / Z
L   = m + log(Z)
```

This implementation's cluster merge uses `expf` and `logf`. Keep the logarithm convention consistent between partial attention, cluster merging and the final combine. The [original Flash-Decoding description](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) explains the split-attention and final-reduction structure.

## Cluster communication and software reduction

The ordinary path writes each split's output and LSE to global scratch, then launches a combine kernel. The experimental path detects groups of splits that belong to the same request and merges part of that work within a CTA cluster:

```text
partial attention in each CTA
    -> shared output vectors and LSE
    -> DSM exchange within an eligible cluster
    -> FP32 weighted merge on row-owning CTAs
    -> final output or a smaller set of global partials
    -> final combine for remaining splits
```

The TMA helper emits:

```ptx
cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes
```

This instruction copies data. The CUDA threads calculate the LSE weights and FP32 weighted sum. This forward path should not be described as using `cp.reduce.async.bulk` to compute attention reduction arithmetic.

Output vectors move through bulk shared-to-shared transfers. LSE values are exchanged using separate DSM stores. A row has 512 logical FP32 output elements, with a physical stride of 520 floats in the updated output and receive layouts. Each output-row transfer copies `kHeadDimV * sizeof(ElementAccum)`, or 2048 bytes for the current 512-dimensional specialization. Unlike the later sibling implementation, this version does not pack the LSE into that output transfer.

The exchange reuses attention shared-memory storage and partitions row ownership across CTAs and warps. It uses cluster synchronization and transaction-barrier objects, but some expected-byte accounting is commented out and the visible exchange path needs a careful completion/lifetime audit. A cluster barrier should not be assumed to replace the completion protocol required by an asynchronous transaction. Review publication of the separate LSE stores as well as output-buffer reuse. NVIDIA's [CUDA programming guide](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#distributed-shared-memory) describes the relevant DSM lifetime and TMA synchronization requirements.

The host launch function still launches the final combine kernel after the split kernel. Compressed split counts and direct-final-output cases can reduce its work; this is not an unconditional removal of the combine launch.

## Existing tuning changes

| Area | Preserved working-tree change | Interpretation |
|---|---|---|
| Metadata overhead | `fixed_overhead_num_blocks` changes from 5 to 3. | Alters scheduler payload estimates; requires workload-specific validation. |
| Host partition budget | Device SM query is replaced by the constant 112. | Experimental hardware assumption, not portable device discovery. |
| Small-workload cluster budgets | Both metadata and launcher use 112 instead of 120. | Keeps this pair of edits aligned, but does not validate the full scheduling formulas. |
| Shared-memory row layout | Output and receive row strides change from 512 to 520 floats. | Padding experiment associated with bank-conflict tuning; buffer capacity and indexing still require validation. |

The effective-workload heuristic is based on `batch_size * (132 / num_sm_parts)`. It selects cluster sizes 8, 4, 2 or 1 at thresholds 4, 16 and 64, and rounds the partition count to a cluster-size multiple. The host's fixed 112 budget and the retained 132-based formulas must be evaluated together; do not infer the actual hardware SM count from these constants.

The metadata marks eligible same-request groups with bit 16 of the split metadata and compresses the split counts. Groups crossing request boundaries need the ordinary staging path. Validation must cover both routes, rather than only a shape that always forms complete eligible clusters.

## Hardware and environment continuation

The current server reports RTX 6000D GPUs with compute capability 12.0. The source checks for SM90, and `setup.py` targets `compute_90a`/`sm_90a`. NVIDIA documents that architecture-specific `compute_90a` PTX is not forward-compatible with Blackwell in the [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html).

Porting therefore requires more than changing a compile flag or removing the API check. Review WGMMA use, shared-memory capacity, cluster occupancy, launch attributes, and scheduling assumptions against the target device. The [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) provides resource constraints that differ from Hopper.

At the initial H100 source-preservation snapshot, `~/nhl_project/nhlpy3` used Python 3.12.3 without PyTorch, and copied extension filenames included CPython 3.11 builds. No environment changes were made in that preservation commit. The subsequent RTX port installed the scoped environment described in [RTX6000D.md](RTX6000D.md); follow [the current workflow](WORKFLOW.md) for today's setup. The copied H100 binaries still do not establish a compatible Python 3.12 H100 build. All further setup must remain under `~/nhl_project`.

## Validation needed before performance claims

1. Restore reference computation and output/LSE assertions in the test harness. The currently configured script calls the extension while its comparisons and timing are commented out.
2. Check the 520-stride layouts against every overlaid shared-memory buffer's capacity, alignment, row indexing and stage offsets.
3. Audit async completion, separate LSE visibility, barrier initialization, peer participation and shared-memory lifetime.
4. Exercise cluster sizes 1, 2, 4 and 8; incomplete groups; cross-request groups; multiple head counts; variable lengths; partial tiles; and empty or fully masked work where supported.
5. Check metadata partition counts against launched CTAs for every supported shape. Replace hard-coded hardware budgets only in a separately reviewed implementation change.
6. Measure matched baseline and optimized revisions, recording GPU, compiler, Python/PyTorch versions, imported extension path, dtype, exact shapes, cluster configuration and dirty source state.
7. Report final combine duration, attention-kernel subtotal, full operator latency and model latency separately. Include regressions and measurement spread. Modeled algorithmic bandwidth is not a direct HBM counter measurement.

The wider workspace contains historical profiles, but their labels and timestamps are insufficient to prove that they were generated from this precise working tree. No numerical speedup from another implementation is assigned to this revision.

## Source navigation

Line references describe the preserved source snapshot; later edits may move them.

| File | Relevant locations |
|---|---|
| [API](../csrc/flash_api.cpp) | Lines 23-40: overhead and fixed SM budget; 75-77: SM90 check; 189-196: specialization and dispatch. |
| [Metadata](../csrc/flash_fwd_mla_metadata.cu) | Lines 30-44: cluster heuristic and budget; 90-137: work assignment and compressed groups. |
| [Kernel](../csrc/flash_fwd_mla_kernel.h) | Lines 153-166: S2S helper; 508-615: shared output/receive layouts; 617-682: exchange; 685-780: weighted merge and output. |
| [Launcher and combine](../csrc/flash_fwd_mla_kernel.h) | Lines 985-1034: cluster choice, launch and final combine; 1038-1043: fixed kernel specialization. |
| [Test entry point](../tests/test_flash_mla.py) | Lines 110-125: disabled reference checks and timing; 128-145: active shape. |
| [Build configuration](../setup.py) | Lines 42-46: submodule setup and SM90a target; 58: extension identity. |

The unchanged [upstream README](UPSTREAM_README.md) retains the original project's installation, performance and citation context. Those upstream statements are not new measurements of this experimental revision.
