# Larger-row and sequence-length results on RTX 6000D

Completed 26 October 2026. The frozen `constant S=4` policy achieves **1.259x** pooled full-pipeline geometric speedup over the unchanged production heuristic on 12 fresh held-out cases: **1.259x** in B and **1.259x** in C. However, `N01_full` regresses by **12.32% / 12.35%** in B/C. The policy remains offline.

This is an offline split-policy experiment with unchanged kernels and production
defaults. It extends the [25 October study](SM120_SPLIT_POLICY_RESULTS.md), whose
held-outs were already observed. The new shape/capacity combinations and tensor
seeds were fixed before training; none duplicates a version 1 case. The
[frozen method](SM120_LENGTH_STUDY_METHOD.md) defines all choices and gates.

## Coverage, acquisition and correctness

- 24 BF16 causal cases: four training and four held-out shape families, each
  with full, quarter and skew profiles; 4,096–8,192 output rows and 256–512
  producer groups. Allocated capacities range from 4,096 to 32,768 tokens;
  held-out capacities are 6,144, 12,288, 24,576 and 20,480. Quarter/skew pairs
  share total live tokens and capacity; exact lengths are in the manifest.
  Sq is 4 or 8, extending query-row count beyond single-query decoding; this
  purposive matrix is not a measured production serving distribution.
- A measures 12 training cases. B/C each measure the 12 held-out cases in a
  fresh process. All three sessions use one idle RTX GPU at a time, nhlpy3,
  Torch 2.9.1+cu128, Triton 3.5.1,
  CUDA 12.8, 156 SMs. No package installation or
  global change was made. Raw working files remain outside the checkout.
- **252 candidate-runs** passed independent FP32 eager and graph-replay
  checks, plus **42** untimed empty/masked preflight configurations across
  three sessions. All 36 public-default pipelines were bitwise equal in eager
  and graph mode. Existing numerical tolerances were unchanged.
- Seven choices S={1,2,4,8,9,16,32}, 21 balanced rounds per stage, 128 graph
  calls per observation. These are warm preallocated CUDA Graph measurements,
  not public-call overhead, cold-cache, full-model or serving timings.
- Before/after source hashes and five-second compute-process monitoring passed.
  All 11 H100 CUDA files remain identical to preservation commit
  `10c778c1f1673c344fe57447447653e4a0c623ee`; RTX runtime SHA256 remains
  `764f799a5ac19e9f48c35a1ba64efd7d05fb03d60cede78a47263838dabaedf1`.
  This does not establish H100 runtime correctness or performance.

## Training-only selection

Leave-one-shape-family-out CV keeps all three length profiles together. All
candidate splits of a case stay in the same fold. The selected model family is
`constant`. Final-refit nonconvergence fallback:
`False`. No B/C result entered selection or fitting.

| Family | Lambda | Grouped-CV GM | Worst latency/default | Converged | Eligible |
|---|---:|---:|---:|---|---|
| default | - | 1.0000x | 1.0000 | True | True |
| constant | - | 1.1790x | 1.0455 | True | True |
| affine | 0.001 | 0.9190x | 1.8343 | True | False |
| affine | 0.01 | 0.9190x | 1.8343 | True | False |
| affine | 0.1 | 0.8848x | 1.8343 | True | False |
| affine | 1.0 | 0.9668x | 1.5241 | True | False |
| physics | 0.001 | 1.0920x | 1.8343 | True | False |
| physics | 0.01 | 1.0920x | 1.8343 | True | False |
| physics | 0.1 | 0.9689x | 1.8343 | True | False |
| physics | 1.0 | 0.9308x | 1.5836 | True | False |

The selected family's worst training fold is `L01_skew`:
S2 -> S4, with
4.55% higher latency. This regression
is retained, even though the geometric CV score passes the eligibility gate.

The eligibility gate requires >=1% geometric gain, <=10% worst case slowdown,
and converged fitting. All new held-outs are inside the expanded marginal
support bounds; this does not imply every feature interaction is well modeled.
Frozen predictions are identical across the three profiles within each family,
because the model reads only host-visible shape and capacity.

## Fresh held-out full-pipeline results

| Case | Default S | Policy S | Grid best B / C | Default B / C (us) | Policy B / C (us) | Speedup B / C |
|---|---:|---:|---|---|---|---|
| N01_full | 2 | 4 | 1 / 1 | 922.461 / 922.306 | 1036.135 / 1036.228 | 0.890x / 0.890x |
| N01_quarter | 2 | 4 | 1 / 1 | 239.769 / 239.490 | 243.608 / 243.512 | 0.984x / 0.983x |
| N01_skew | 2 | 4 | 8 / 8 | 306.567 / 306.760 | 307.638 / 308.154 | 0.997x / 0.995x |
| N02_full | 1 | 4 | 2 / 2 | 3376.423 / 3375.951 | 2883.368 / 2882.437 | 1.171x / 1.171x |
| N02_quarter | 1 | 4 | 2 / 2 | 862.950 / 863.621 | 775.029 / 775.220 | 1.113x / 1.114x |
| N02_skew | 1 | 4 | 4 / 4 | 1527.983 / 1526.924 | 814.754 / 814.821 | 1.875x / 1.874x |
| N03_full | 1 | 4 | 9 / 9 | 6705.164 / 6703.881 | 5027.605 / 5028.305 | 1.334x / 1.333x |
| N03_quarter | 1 | 4 | 9 / 9 | 1522.356 / 1525.962 | 1296.633 / 1296.732 | 1.174x / 1.177x |
| N03_skew | 1 | 4 | 4 / 4 | 2594.311 / 2592.999 | 1304.582 / 1304.746 | 1.989x / 1.987x |
| N04_full | 1 | 4 | 4 / 4 | 7439.423 / 7438.461 | 6272.913 / 6273.080 | 1.186x / 1.186x |
| N04_quarter | 1 | 4 | 4 / 4 | 1894.034 / 1893.184 | 1638.233 / 1638.192 | 1.156x / 1.156x |
| N04_skew | 1 | 4 | 4 / 4 | 2978.281 / 2978.269 | 1708.459 / 1708.712 | 1.743x / 1.743x |

Practical outcomes (threshold max(0.1us,1% of default)): B
`{'loss': 2, 'tie': 1, 'win': 9}`; C `{'loss': 2, 'tie': 1, 'win': 9}`.
Pooled profile geometric speedups: full **1.133x**,
quarter **1.105x**, skew **1.595x**.
The exploratory 95% family/session bootstrap interval is
**1.041–1.429x**.
Only two process sessions and four held-out families support this interval.
Matched profiles and repeated graph calls do not increase that independent
sample count. Raw per-choice medians, MAD/IQR, ranges and round samples remain
in the [archive](data/sm120_length_study_20261026/).

## Matched total-token imbalance

Ratios below compare skew to uniform quarter-length at the same shape,
capacity, candidate policy and exact total live tokens. A ratio above 1 means
skew is slower. This comparison changes distribution and maximum live length;
it does not hold each sequence's length fixed.

| Family | Equal total live tokens | Default skew/quarter B / C | Policy skew/quarter B / C | Best S quarter B / C | Best S skew B / C |
|---|---:|---|---|---|---|
| N01 | 9216 | 1.279x / 1.281x | 1.263x / 1.265x | 1 / 1 | 8 / 8 |
| N02 | 36864 | 1.771x / 1.768x | 1.051x / 1.051x | 2 / 2 | 4 / 4 |
| N03 | 61440 | 1.704x / 1.699x | 1.006x / 1.006x | 9 / 9 | 4 / 4 |
| N04 | 81920 | 1.572x / 1.573x | 1.043x / 1.043x | 4 / 4 | 4 / 4 |

These are separate captures with manifest-order profiles, not one captured
graph replayed after lengths mutate. Existing runtime graph tests cover that
behavior separately; no new changed-length capture claim is made here.

## Producer and finish cost

Session B stage medians explain tradeoffs; C and all raw samples are archived.
Full latency above is measured directly and must not be reconstructed by
adding these separately acquired stage medians. Finish-only measurements read
resident partial buffers, whereas full includes producer writes.

| Case | Default producer / finish (us) | Policy producer / finish (us) | Grid best producer / finish (us) | Policy regret to grid best |
|---|---|---|---|---:|
| N01_full | 906.089 / 5.654 | 1027.712 / 7.261 | 908.038 / 2.914 | 13.78% |
| N01_quarter | 233.894 / 5.865 | 235.698 / 7.695 | 231.733 / 3.315 | 3.66% |
| N01_skew | 300.932 / 5.644 | 299.923 / 7.281 | 271.601 / 13.616 | 2.37% |
| N02_full | 3362.760 / 3.424 | 2843.527 / 9.334 | 2807.873 / 7.223 | 1.89% |
| N02_quarter | 856.930 / 3.414 | 754.507 / 9.341 | 729.710 / 7.231 | 4.75% |
| N02_skew | 1519.263 / 3.393 | 787.577 / 9.360 | 787.577 / 9.360 | 0.00% |
| N03_full | 6698.967 / 3.079 | 4997.595 / 7.991 | 4723.259 / 16.540 | 4.90% |
| N03_quarter | 1495.602 / 3.080 | 1282.392 / 8.010 | 1213.068 / 16.539 | 2.29% |
| N03_skew | 2583.830 / 3.074 | 1282.123 / 8.009 | 1282.123 / 8.009 | 0.00% |
| N04_full | 7420.400 / 4.093 | 6216.276 / 12.092 | 6216.276 / 12.092 | 0.00% |
| N04_quarter | 1879.086 / 4.095 | 1589.321 / 12.091 | 1589.321 / 12.091 | 0.00% |
| N04_skew | 2959.131 / 4.092 | 1652.089 / 12.092 | 1652.089 / 12.092 | 0.00% |

## Grid headroom and previous-policy comparison

The best sampled S agrees across B/C on **12/12** cases. Same-session
grid headroom is **1.294x** in B and
**1.295x** in C. These minima are selected
from seven choices, not a global optimum. Independent-session selectors give:

- B -> C: default/selector **1.295x**; policy/selector **1.029x** (values above 1 mean the policy is slower).
- C -> B: default/selector **1.294x**; policy/selector **1.027x** (values above 1 mean the policy is slower).

The previous frozen v1 policy falls back to production for all new cases
because its row/group training support is smaller. Its guarded speedup is
therefore 1.000x. Without that guard, its offline diagnostic geometric speedup
would be **1.080x** in B and
**1.081x** in C. This ablation
is not permission or evidence to remove the guard. Per-case old choices and
fallback reasons are retained in the frozen record and evaluation.

Its worst unguarded case is `N02_quarter` at
S32: latency increases
**51.30% / 51.19%**
in B/C relative to production. The aggregate improvement must not hide this
failure of model extrapolation.

## Interpretation and next experiment

The benefit in this expanded domain comes from the joint producer/finish
tradeoff. The separately timed finish is slower with S4 in every held-out
case, by a geometric factor of **2.282x** across sessions, while
the full pipeline improves geometrically. This is a scheduling gain despite
more finish work, not a new faster reducer kernel.

For N02-N04, the skew profile improves **1.743–1.989x**
across the two sessions. Their matched-token tables show how a fixed total
token count can conceal distribution-dependent producer cost. Capacity alone
does not describe that distribution. This is evidence to investigate length
imbalance and shape interactions, not proof of a particular SM-utilization
or memory-bandwidth mechanism: those counters were not measured.

For a concrete geometry example, N04_skew has 15 sequences of length 4,096
and one of length 20,480, versus 16 sequences of 5,120 in its quarter profile.
Both have 81,920 live tokens and the same number of 32-token blocks. The long
sequence has 32 producer groups, so S1 gives it 32 producer CTAs with 640
iterations each; S4 gives it 128 CTAs with 160 iterations each. These counts
come from the unchanged [producer loop](../flash_mla_old/sm120.py#L40-L46)
and launch formulas, not profiler measurements.
They motivate a hypothesis about distributing a long sequence across more
CTAs; an actual scheduling/occupancy explanation still requires hardware data.

N01_full regresses in both sessions despite lying inside the expanded marginal
support bounds. Its 96 query heads, 48 heads per KV head and 288 producer
groups form an unseen interaction. These observations motivate targeted
characterization, but do not isolate which dimension causes the regression.
Do not add a special-case dispatch rule from this held-out outcome. The old
unguarded model's larger regression also shows why extending bounds alone
does not repair ranking errors.

The separately measured resident-finish/full ratio ranges from
**0.05% to 3.16%** for default and S4 on these
held-outs. It is not an additive critical-path fraction or an Amdahl bound.
It suggests prioritizing full-pipeline scheduling in this larger-row domain,
while investigating reducer mechanisms where finish cost is actually exposed.
The original low-row and Hopper/TMA hypotheses remain separate experiments.

The next protocol should cover both small and large rows, compare default,
constant S4 and a new model on untouched families, and account for metadata
availability and policy overhead. Any live-length feature needs a declared
graph-compatible path without hidden CPU reads of CUDA lengths. FP16,
changed-length replay within one capture, public-call overhead and layer-level
validation remain open. The present two-session result supports further
research, not a blanket production split-policy change.

Do not refit using these now-observed held-outs or promote the policy to
production from this result. A next study needs new families, predeclared
features and a fresh freeze. Distinguish full producer-plus-finish costs from
reducer-only speed. No TMA, SM100 DeepInfer, energy-per-token or model-level
benefit was measured, and no architecture conclusion should rely on those
unmeasured quantities.

## Evidence and reproduction

- [Protocol](../benchmarks/sm120_length_study_protocol.json), SHA256
  `232a2c87398ee0a22e7f85769ca3f15787d9ad2df8923c5a3ee405eaa30146b5`.
- [Frozen policy](data/sm120_length_study_20261026/frozen_policy.json), SHA256
  `e24b35edaf7e39bba9ac8c6dda24f8ced36399671ed4dde4b36d82d047ae8fb6`; created `2026-10-26T05:53:09.723762+00:00` after A and before B/C.
- [Preregistration](data/sm120_length_study_20261026/preregistration.json),
  frozen `2026-10-26T05:37:59Z` against base
  `16fa2f3fc16e044b45576c4d2422cb6d2ac6cc12` before training timings.
- [CPU analyzer](../benchmarks/analyze_sm120_length_study.py) reuses version 1
  model primitives but changes CV to family groups and evaluates held-out-only
  sessions. [CPU checks](../tests/test_sm120_length_analysis.py) cover leakage,
  matched tokens, support and default tie behavior.
- [Operation steps](WORKFLOW.md#9-reproduce-the-larger-row-and-matched-length-study)
  and [data/hash index](data/sm120_length_study_20261026/README.md).

The experiment used physical GPU 0
for A, 0 for B and
0 for C. GPU guards are periodic
observations, not reservations; warm resident data, deterministic order, a
purposive workload set, BF16 only and a single available architecture limit
generalization. Version 1 evidence and original kernels are preserved.
