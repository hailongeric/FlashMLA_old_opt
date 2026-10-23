# Hopper research readiness and evidence audit

Audit date: 24 October 2026 (Asia/Shanghai). Only the RTX 6000D server is
available. The H100 implementation is preserved, but no new Hopper execution,
correctness, performance or energy result is claimed.

## Current DeepInfer is a separate integration target

The [24 October DeepInfer audit](DEEPINFER_APPLICABILITY.md) traces the official
framework and its actual FlashMLA gitlink. Non-fused NVIDIA decode retains a
separate combine when split-KV is enabled; fused decode explicitly disables
split-KV. The fused implementation already uses TMA and CTA clusters. There
is no universal standalone combine to replace across current model layers.

Porting the hierarchical idea requires new SM100 scheduling, quantized-cache
and sink/LSE contract work. It cannot reuse the preserved SM90 kernel as a
drop-in extension. Current upstream validation needs compatible SM100-family
hardware; future H100 access would validate the historical experiment only.
RTX measurements validate neither upstream target. No environment or kernel
change was made as part of the upstream source audit.

## What is implemented

The preserved `FlashMLA_old_opt` kernel merges partial attention outputs within
a CTA cluster, inside the split-KV producer. It redistributes shared-memory
output rows with `cp.async.bulk.shared::cluster.shared::cta`, then performs
the LSE-rescaled weighted sum in software. A final global combine kernel is
still launched. This is hierarchical reduction, rather than a standalone
global-memory combine kernel converted to TMA.

The separate RTX backend uses Triton attention and reduction. Its current
reduction policy and direct-output path do not implement the Hopper cluster
exchange. The [new RTX measurements](RTX6000D_LOW_BATCH.md) therefore support
RTX characterization and model development only.

The 11 tracked CUDA source files under `csrc/`, excluding CUTLASS, remain
byte-identical to preservation commit
`10c778c1f1673c344fe57447447653e4a0c623ee`. The audit used checkout
`fe1620954d22cd2f81650cbae073570fcd2e79c4`. See the earlier
[TMA design record](TMA_REDUCTION.md) for the implementation map.

## Findings that must precede Hopper performance claims

### 1. Single-split condition has a confirmed precedence defect

At line 546 of `csrc/flash_fwd_mla_kernel.h`, the expression is:

```cpp
END_SP_IDX + n_split_idx & 0xffff - split_idx == 1
```

C++ parses it as:

```cpp
(END_SP_IDX + n_split_idx) & ((0xffff - split_idx) == 1)
```

For `END_SP_IDX=1`, `n_split_idx=0`, `split_idx=0`, the current condition is
false; `END_SP_IDX + (n_split_idx & 0xffff) - split_idx == 1` is true.
An isolated CPU program compiled with the installed `g++ -Wall -Wextra`
confirmed this parse and six contrasting examples. This checks expression
semantics; it does not validate the proposed condition against the metadata
algorithm or prove the output behavior of a compiled Hopper kernel.

The earlier assignment at line 498 also masks the sum:
`(params.num_splits_ptr[bidb] + n_split_idx) & 0xffff`. Audit its intended
offset range and packed-field meaning along with the condition. A patch must
be validated on actual metadata cases before adoption. Both expressions were
left unchanged to preserve the user's kernel work.

### 2. DSM copy completion is not established by the visible protocol

The helper at lines 153–166 issues a nonblocking bulk copy tied to an
`mbarrier`. The exchange at lines 634–682 initializes `barriers_dsm`, issues
copies at lines 651 and 665, and uses `cluster.sync()`. The corresponding
`arrive_and_expect_tx` calls are commented out, and there is no wait on these
DSM transaction barriers before the consumer reads the received data. This
is a source-level synchronization concern requiring a corrected protocol
and Hopper runtime validation; the audit did not observe a GPU race directly.

NVIDIA specifies mbarrier-based completion for this shared-to-cluster copy
variant. A cluster rendezvous alone does not establish completion of that
asynchronous transaction. Review expected-byte accounting, barrier location,
phase/lifetime, async-proxy visibility, and source-buffer reuse together.
[NVIDIA PTX bulk-copy specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)

Do not merely uncomment one arrival: the full producer/consumer protocol and
stage reuse need to agree. Preserve a distinct candidate revision for such
changes and compare it to an independent attention oracle on Hopper.

### 3. Numerical and scheduling coverage is incomplete

The historical `tests/test_flash_mla.py` contains disabled reference checks.
The saved `tests/log` records shapes but does not establish correctness,
source revision or timing. Current RTX tests exercise the RTX arithmetic and
mocked architecture dispatch; they do not validate Hopper execution.

Hopper tests must cover cluster sizes and cross-request fallback, partial
head tiles, uneven lengths, fully masked/empty splits, single remaining
split, and multiple remaining splits. Check FP16 and BF16 output, FP32
accumulation, natural-log public LSE, and internal base-2 LSE separately.
Investigate the historical positive-infinity sentinel before applying the
RTX empty-row contract mechanically to the Hopper implementation.

## Historical evidence inventory

Three reports were retained unchanged in the checkout's untracked `tests/`
directory:

| Report | Bytes | SHA-256 |
|---|---:|---|
| `mla_old_opt_b1h128.ncu-rep` | 4620753 | `b283ceddfb8d3d9cc3c1356ba3d008928b4ff4fc646750d05c82939a3f928ebe` |
| `mla_old_opt_b1h128_2.ncu-rep` | 4562781 | `44383daa55b7e3a1fd19f68a43c8b16f003aba5a306c7ffd3c0ade75cda77ef5` |
| `mla_old_opt_b1h128_3.ncu-rep` | 4534265 | `22cabc0daef1ebf9ffd090e0ad374d631368c9eaf49b776caf4bd7edc6cd44003` |

Installed Nsight Compute is **2022.4.1.0**; Nsight Systems is **2022.4.2**.
Read-only `ncu --import ... --page raw --csv` failed for all three reports
with missing required protobuf fields in `ProfileResult.RuleResults`.
No usable device, kernel timing or counter rows were extracted. This is a
reader/report compatibility failure; it is not proof the original reports
are corrupt. No profiler was installed or upgraded.

The sibling file `kernels/FlashMLA/benchmark/paper_data/mla_kernels.csv`
contains historical combine ratios for several implementation labels and
shapes. It does not bind them to this repository's exact source/build/device,
and its rows have more fields than the header. It is a lead for provenance
recovery, not a verified performance baseline. The inspected
`benchmarks/vllm_test/old/power_benchmarking.csv` contains only a header and
supplies no energy samples.

Raw import diagnostics, file hashes, the CPU expression probe and new RTX
runs are saved outside the Git checkout at
`~/nhl_project/moore_tma_pj/docs/research_preparation_20261024/`.
The historical profiler binaries were not committed or modified.

## Reproducible Hopper validation protocol

These steps are pending hardware access. They are not completed experiments.

1. Reserve an available H100/H800, record device/driver/toolchain, and bind
   the imported extension to an exact source commit and binary hash. Use a
   compatible project-local environment; do not replace shared dependencies.
2. Select a baseline from the same kernel/API generation. Local history
   `a2d0c78` is labeled as an original version, but that label alone does not
   establish a fair baseline. Audit it before building. Upstream's historical
   Hopper revision `ba89a3466e9470ad08ab39738d4e7bb66989e1e7` is another
   reference, not an automatic drop-in control.
3. Resolve the expression and synchronization findings in a separate
   candidate. Run independent FP32 attention checks, repeated stress cases,
   CUDA Graph replay with changing live lengths, and available memory/race
   diagnostics before collecting performance data.
4. Measure an unfused reference, synchronous DSM exchange if available, and
   asynchronous DSM/TMA exchange with the same scheduling and math. Sweep
   batch, context length, legal split count and cluster size. Record producer,
   residual combine and full-operator latency separately.
5. Collect counters with a compatible existing profiler if available. Define
   exactly which SM-active/throughput metric is reported, whether traffic is
   L2 or HBM, and how profiling replay differs from unprofiled timing.
6. For energy, measure a sufficiently long repeated workload with synchronized
   duration and sampled/integrated power, document idle treatment and token
   count, and repeat trials. A single `nvidia-smi` power reading is not energy
   per token. Full-model decoding needs a separate end-to-end experiment.

The [pinned upstream release](https://github.com/deepseek-ai/FlashMLA/commit/2e5429fc5653bab6e081f09477126f731882a6a9)
records removal of Hopper support on 30 September 2026. Use a historical,
API-compatible Hopper baseline. The current framework/dependency/main pins
and source hashes are in [the DeepInfer audit](DEEPINFER_SOURCE_AUDIT.json).

## Research model and novelty gates

For R output rows, V=512 and S partials, logical FP32 scratch is
`4*R*S*(V+1)` bytes. A cluster of C fully coalescible splits could reduce
the global partial count toward `ceil(S/C)`; cross-request boundaries,
scheduler layout and partial clusters can reduce that benefit. This is a
logical traffic model, not measured HBM traffic.

A useful break-even hypothesis is:

```text
saved global staging and residual-combine time
    > DSM transfer + transaction synchronization + occupancy/producer penalty
```

Measure every term or calibrate it transparently. The producer changes in
the hierarchical design, so a combine-only Amdahl calculation cannot predict
full-operator speedup by itself. The [RTX report](RTX6000D_LOW_BATCH.md)
includes a separate, exploratory bytes-only fit with held-out split counts;
those coefficients do not transfer to Hopper.

Meld is a direct prior-art risk: its author repository describes SM
cooperation and just-in-time reduction with cluster communication. Compare
the reduction ownership, readiness protocol, scheduling, supported layouts
and measured tradeoffs before claiming novelty. The supplied 63%/37% timing
split and 10.3% utilization numbers remain unverified for a specific workload
and metric definition.
[Meld author repository](https://github.com/He-Jingkai/tilelang.dataflow),
[SOSP 2026 accepted papers](https://sigops.org/s/conferences/sosp/2026/accepted.html)

ThunderMLA explores a megakernel route. Treat compatibility with it as a
hypothesis until an integrated implementation is measured; sharing a design
goal is not evidence that speedups compose.
[ThunderMLA authors' description](https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla)

With RTX-only access, the concrete next research result is portable reduction
characterization and a validated model of dispatch/traffic/launch costs.
A Hopper TMA performance claim remains gated on Hopper hardware and the
correctness work above.
