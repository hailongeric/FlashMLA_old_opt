# RTX 6000D split-policy study: 25 October 2026

## Outcome

A cost model fitted on 12 predefined workloads selected split counts that achieved **1.207x geometric-mean full-pipeline speedup** over the unchanged production heuristic on eight held-out workloads across two fresh-process sessions. Session B measured **1.210x** and session C **1.205x**. Restricting the summary to the seven cases inside the training support gives **1.240x** pooled speedup. H08 deliberately falls outside support and retains the default.

These are warm, preallocated CUDA Graph operator timings on the existing RTX implementation. The model is an offline research artifact; **no production policy or kernel was changed**. This is not Hopper TMA, DeepInfer, model-serving, or energy evidence. The policy comparison uses the already optimized RTX backend, whereas the baseline confirmation below compares that backend with the original RTX port. Do not multiply the two speedups or describe them as one matched experiment.

The [method](SM120_SPLIT_POLICY_METHOD.md), [protocol](../benchmarks/sm120_split_policy_protocol.json), [frozen model](data/sm120_split_policy_20261025/frozen_policy.json), [evaluation](data/sm120_split_policy_20261025/evaluation.json), and [raw-data index](data/sm120_split_policy_20261025/README.md) make the study reviewable. The protocol, source files, selection rules and workload split were recorded before new timings. The fitted model and all held-out predictions were frozen before B/C. This local audit trail establishes artifact identity and ordering; it is not an external preregistration service.

## Corrected confirmation of the existing optimization

Review found that the previous timing helper rotated names and reversed odd rounds. With two functions, those actions cancelled, so every round ran the baseline first. The old results remain unchanged and are labeled with this limitation. The helper now reverses alternate complete rotation cycles. The corrected forward run uses 20 rounds to balance both its five-function and two-function groups; the synthetic reducer uses 16 rounds for its two-function group.

| Measurement session | Full forward, 24 cases | Retained reducer, 16 cases | Synthetic multi-split reducer, 28 cases |
|---|---:|---:|---:|
| 24 October, original method | 1.099x | 2.085x | 1.694x |
| 25 October, same-method repeat | 1.099x | 2.088x | 1.693x |
| 25 October, balanced order | 1.101x | 2.102x | 1.677x |

All 24 corrected full-forward medians improved relative to the original port. The synthetic R=128, S=1 copy regression remains: **1.165 to 1.207 us**, or **3.63% higher latency**. Retaining it is necessary even though the short-capacity production path bypasses this copy. Small differences remain timing observations, not statistical guarantees. The same-method repeat alone would not fix the ordering limitation.

Both implementations passed the independent FP32 attention oracle in the 24-case forward confirmations and the FP64 weighted-reduction checks in the 32-case synthetic confirmations. The corrected full-forward run used physical GPU 0; the synthetic run used GPU 1, each acquired idle. Their ratios compare baseline and candidate on the same selected device within each run.

## Frozen design and correctness

The manifest fixes BF16, causal attention, QK576, V512 and 64-token pages. Workloads vary batch, query length, query/KV head grouping, capacity and heterogeneous live lengths. Twelve cases train the model; eight are held out. Every case has seven candidate split counts including the observed production choice. Candidate counts beyond the heuristic are legal and may produce empty partials.

Session A includes training cases only. B/C each rerun all 20 cases in a fresh process, retaining identical seeded inputs to isolate session variation. Training rows in B/C are transfer checks and never refit the model. Full, producer and finish timings are separate groups. Requested 15 rounds become 21 for seven candidates or 18 for a six-candidate finish group; every candidate occupies every position equally. Each observation contains 128 graph calls, not 128 independent samples. Full latency is measured directly, not reconstructed by adding stage medians.

Across A/B/C, **364 candidate-runs** passed eager and graph replay against the independent attention reference. The production-selected pipeline also matched the public API bit-for-bit in both modes on all 52 case-runs. An additional **76 untimed preflight configurations** checked empty and fully masked rows in eager and graph execution, covering direct, copy and merge routes. Existing tolerances were retained. All sessions completed with unchanged source hashes and no foreign compute process observed by the guards. These sampled checks do not reserve a GPU or exclude every possible source of interference.

| Session | Phase | Cases | Started UTC | Completed UTC | GPU | Process |
|---|---|---:|---|---|---|---:|
| A | train | 12 | 2026-10-25T05:53:54Z | 2026-10-25T05:58:09Z | 0 | 3703139 |
| B | all | 20 | 2026-10-25T05:58:38Z | 2026-10-25T06:06:02Z | 0 | 3706615 |
| C | all | 20 | 2026-10-25T06:06:05Z | 2026-10-25T06:12:41Z | 0 | 3710645 |

The stack remained nhlpy3, PyTorch 2.9.1+cu128, Triton 3.5.1, SM120 with 156 SMs. JIT compilation, allocations, metadata, H2D, numerical reference and policy Python execution are outside the timed graphs. Processes reuse shared JIT caches. These are separate acquisition intervals on one server, not independent hardware trials.

## Held-out full-pipeline results

Latencies are medians in microseconds. Speedup is default divided by the frozen policy. Exact ties where the split is unchanged use the same candidate samples; they are not independent A/B kernel measurements. The grid-best columns describe the lowest measured median among seven choices, not a global optimum.

| Case | Default S | Policy S | B default us | B policy us | B speedup | C speedup | Best S B/C | Practical result B/C |
|---|---:|---:|---:|---:|---:|---:|---|---|
| H01 | 12 | 32 | 20.989 | 10.795 | 1.944x | 1.942x | 32/32 | win/win |
| H02 | 13 | 13 | 78.538 | 78.538 | 1.000x | 1.000x | 13/13 | tie/tie |
| H03 | 7 | 16 | 501.706 | 399.357 | 1.256x | 1.255x | 16/16 | win/win |
| H04 | 5 | 16 | 1111.098 | 1024.608 | 1.084x | 1.072x | 32/32 | win/win |
| H05 | 26 | 26 | 47.674 | 47.674 | 1.000x | 1.000x | 32/32 | tie/tie |
| H06 | 7 | 16 | 333.051 | 226.763 | 1.469x | 1.445x | 16/16 | win/win |
| H07 | 7 | 16 | 248.076 | 210.198 | 1.180x | 1.175x | 32/32 | win/win |
| H08 | 2 | 2 | 1552.147 | 1552.147 | 1.000x | 1.000x | 9/9 | tie/tie |

Practical wins/losses require an absolute change exceeding `max(0.1 us, 1% of default latency)`. No held-out practical losses occurred in either session.

H08 has more rows and producer groups than the training support and intentionally falls back to S=2. Its missed grid headroom remains in regret summaries. H01-H04 have uniform live lengths; H05-H08 are heterogeneous. See the method table for exact shapes and seeds. Marginal support bounds do not establish coverage of every interaction inside them.

| Regret / headroom | Session B | Session C |
|---|---:|---:|
| Default / sampled grid-best, geometric mean | 1.310x | 1.303x |
| Policy / sampled grid-best, geometric mean | 1.083x | 1.082x |

Choosing the grid winner in session B and evaluating it in C gives **1.303x** versus default. The frozen policy's geometric latency ratio to that cross-session selector is **1.082x**. Signed per-case gaps, including any negative values, are retained in the evaluation JSON.

Choosing the grid winner in session C and evaluating it in B gives **1.310x** versus default. The frozen policy's geometric latency ratio to that cross-session selector is **1.083x**. Signed per-case gaps, including any negative values, are retained in the evaluation JSON.

Grid winners agree between B and C on all eight held-out cases. The remaining
gaps are therefore repeatable in these sessions, although they do not establish
global optima. The largest missed opportunities are:

| Case | Policy / best split | B excess latency vs best | C excess latency vs best | Interpretation |
|---|---|---:|---:|---|
| H08 | 2 / 9 | 48.61% (507.73 us) | 47.97% (499.96 us) | Declared support fallback |
| H05 | 26 / 32 | 14.24% (5.94 us) | 13.48% (5.65 us) | Within-support ranking gap |
| H07 | 16 / 32 | 9.20% (17.72 us) | 9.20% (17.71 us) | Within-support ranking gap |
| H04 | 16 / 32 | 1.99% (20.03 us) | 2.21% (22.18 us) | Within-support ranking gap |

The optional paired hierarchical bootstrap gives an exploratory 95% interval of **1.052–1.423x** for the pooled speedup. With only two session clusters and a deliberately chosen small workload set, this is descriptive and is not a robust population confidence guarantee.

## Model selection and stage costs

The chosen nonnegative ridge model is **physics**, lambda **0.001**. Its whole-case leave-one-out training score was **1.288x**, with worst selected/default latency ratio **1.000x**. Families, lambda grid, gain/regression gates, local switching margin and support guard were fixed before A. Only A training measurements selected the model; no B/C measurements changed it.

| Family | Lambda | CV geometric speedup | Worst CV latency ratio | Eligible |
|---|---:|---:|---:|---|
| default | - | 1.000x | 1.000x | yes |
| constant | - | 1.168x | 1.872x | no |
| affine | 0.001 | 1.000x | 1.000x | no |
| affine | 0.01 | 1.000x | 1.000x | no |
| affine | 0.1 | 0.757x | 28.427x | no |
| affine | 1.0 | 1.000x | 1.000x | no |
| physics | 0.001 | 1.288x | 1.000x | yes |
| physics | 0.01 | 1.228x | 1.000x | yes |
| physics | 0.1 | 1.157x | 1.000x | yes |
| physics | 1.0 | 1.157x | 1.000x | yes |

After undoing feature scaling, the fitted cost in microseconds is approximately
`2.197285 + 2.091707 * longest_wave_proxy + 0.039929 * scratch_MiB + 3.215446 * finish_indicator`;
the other fitted nonnegative coefficients are zero in this artifact. The full
coefficient vector, scales and convergence record are retained in the model JSON.

A training-only example shows why reduction latency alone is the wrong objective:
T06 changes S=1 direct output to S=4 with a merge launch. Producer latency drops
from 19.258 to 7.790 us, while the new finish costs
1.662 us. The directly measured full pipeline improves from
19.258 to 9.432 us. This example informed the training fit;
it is not an additional held-out success or a measured SM-utilization claim.

The features use host-visible shapes and allocated capacity, not a GPU-to-CPU read of actual live lengths. The launch-wave terms are scheduling proxies, not measured occupancy or SM utilization. Zero coefficients in a correlated fit do not prove that the corresponding hardware cost is absent.

The following session-B component medians explain the changed held-out choices. They use resident scratch and are diagnostic; producer plus finish medians need not equal the measured full pipeline.

| Case | S default→policy | Producer us default→policy | Finish us default→policy | Full us default→policy |
|---|---|---|---|---|
| H01 | 12→32 | 19.898→9.447 | 1.106→1.371 | 20.989→10.795 |
| H03 | 7→16 | 505.831→396.058 | 2.730→5.147 | 501.706→399.357 |
| H04 | 5→16 | 1082.301→1026.637 | 2.735→6.681 | 1111.098→1024.608 |
| H06 | 7→16 | 328.940→223.451 | 2.780→5.133 | 333.051→226.763 |
| H07 | 7→16 | 245.331→204.936 | 2.771→5.134 | 248.076→210.198 |

## Provenance and next experiments

- Source base: `143b2535c1cdb9218ec23d746704d8fe64c55953`. The benchmark helper change and new research scripts are identified in the [preregistration manifest](data/sm120_split_policy_20261025/preregistration.json).
- Source/protocol freeze: `2026-10-25T05:52:41Z`. Fitted policy freeze: `2026-10-25T05:58:09.958289+00:00`.
- Policy SHA256: `005fed5ad030f2f3cb891b7bb0942a5257a068092b8decf4adaedd8544c4678d`.
- Unchanged RTX runtime SHA256: `764f799a5ac19e9f48c35a1ba64efd7d05fb03d60cede78a47263838dabaedf1`.
- Baseline for the existing-optimization confirmation: `1f168a52a4bce6d34439991ee63070d6a0c49825`. All 11 Hopper CUDA sources remain identical to preservation commit `10c778c1f1673c344fe57447447653e4a0c623ee`.
- Raw logs, source snapshots, backups and GPU-monitor records remain under `~/nhl_project/moore_tma_pj/docs/split_policy_20261025/`. All server work used nhlpy3; no package installation, global setting, clock or power change was made.

The next milestone should expand row/group support and vary live-length skew at matched shapes and capacity, using a separately versioned protocol and fresh untouched workloads. This separates H08's guard-coverage gap from within-support ranking errors such as H05/H07. Include FP16 and inputs whose live lengths change under an existing CUDA Graph. Measure policy/host overhead and a public-API or single-layer boundary before proposing production deployment. An ablation should compare the selected scheduling proxy with simpler fixed heuristics on that new holdout; the rejected constant policy and current affine results already show why average gains alone are insufficient.

Keep H08 and other missed choices as errors to explain, not cases to remove. Any feature or threshold changes motivated by these results require new holdout data. This study does not measure effective HBM bandwidth, occupancy, integrated energy per token, model latency, or causal hardware utilization. GPU telemetry snapshots cannot substitute for those measurements.

The [DeepInfer applicability audit](DEEPINFER_APPLICABILITY.md) remains a source-level integration hypothesis with different SM100 and quantized-cache contracts. The Hopper implementation remains preserved and unvalidated on hardware. Neither receives a speedup claim from this RTX study. Reproduction commands are in [the workflow](WORKFLOW.md#8-run-the-preregistered-fixed-split-study).
