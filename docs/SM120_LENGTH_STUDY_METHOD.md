# Large-row and sequence-length study, version 2

Protocol `sm120-large-row-length-v2-20261026`, specified before new timings on
26 October 2026. The exact manifest is
[`sm120_length_study_protocol.json`](../benchmarks/sm120_length_study_protocol.json).
The preregistration record preserves its hash and all analysis/source hashes.
The [version 1 method](SM120_SPLIT_POLICY_METHOD.md) and its published evidence
remain unchanged. Version 1 outcomes are prior exploratory information;
none of its shape/capacity combinations is reused as a fresh held-out case.

## Question and scope

Does the existing capacity-only split-policy model menu generalize to larger
row/group counts after expanding training support? At fixed shape and allocated
capacity, how does live-length imbalance affect producer, finish, and full
pipeline latency? The primary endpoint is directly measured producer-plus-finish
CUDA Graph latency relative to the unchanged production heuristic.

This is BF16 causal historical dense MLA on RTX 6000D: QK576, V512, page64,
FP32 partials, 156 SMs. It is not an H100 TMA, DeepInfer, energy, or serving
experiment. No kernel, production split heuristic, dependency, device setting,
or H100 source changes. Capacity-only policies cannot distinguish the three
profiles of a shape and must choose the same split count for all three.

## Fresh workload families

Let R=B*Sq*Hq and P=B*Hkv*ceil(Sq*(Hq/Hkv)/16). P counts producer groups before
multiplication by S. Every row below has three profiles, yielding 12 training
and 12 held-out cases. The candidate grid is exactly `{1,2,4,8,9,16,32}`,
including production S1 or S2 for every case.

| Family | Phase | B | Sq | Hq | Hkv | Capacity | R | P |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| L01 | train | 4 | 8 | 128 | 2 | 4096 | 4096 | 256 |
| L02 | train | 8 | 4 | 128 | 2 | 16384 | 4096 | 256 |
| L03 | train | 16 | 8 | 64 | 4 | 8192 | 8192 | 512 |
| L04 | train | 8 | 8 | 64 | 2 | 32768 | 4096 | 256 |
| N01 | holdout | 6 | 8 | 96 | 2 | 6144 | 4608 | 288 |
| N02 | holdout | 12 | 4 | 128 | 4 | 12288 | 6144 | 384 |
| N03 | holdout | 10 | 4 | 128 | 2 | 24576 | 5120 | 320 |
| N04 | holdout | 16 | 4 | 128 | 4 | 20480 | 8192 | 512 |

Profiles are **full** (every sequence equals capacity C), **quarter** (every
sequence equals C/4), and **skew** (last sequence C; distribute the remaining
B*C/4-C tokens as evenly as possible over the first B-1 sequences, with the
remainder assigned one token at a time to the first entries). Thus quarter and
skew have exactly equal total live tokens and identical allocated capacity.
L01 skew includes three empty sequences. Non-page-aligned tails are allowed and
poisoned beyond the valid length by the unchanged harness.

All three siblings share a new per-family tensor/page seed; all families have
different seeds, disjoint from version 1. B/C reuse these declared seeds to
measure session drift. This is fresh workload geometry and input data relative
to version 1, not independent random inputs across B/C. Case order is the
manifest order (full, quarter, skew within each family); profile-order/thermal
effects remain a limitation despite warm-up and balanced candidate order.

## Acquisition and freeze

Reuse the unchanged `benchmark_sm120_split_policy.py`, independent FP32 oracle,
and paired timing helper. Each stage has seven choices, 21 balanced rounds
(requested 15 rounded up), and 128 graph calls per observation. Full, producer,
and finish are timed separately. All cases use the copy/merge route, never the
short-capacity direct route. Finish measurements use resident partials; do not
sum separate medians to estimate full latency.

1. Record protocol, this method, analyzer v2, and unchanged dependency hashes
   before measuring A. A contains only the 12 training cases.
2. Fit and cross-validate on A only, then freeze all 24 predictions, model,
   source/training hashes, and the previous-policy comparator before B starts.
3. B and C each measure the **12 held-out cases only**, in separate fresh
   processes with separate idle-GPU acquisitions and the same frozen policy.
   This deliberately differs from the all-case replication in version 1.
4. Preserve every candidate's eager/graph oracle checks, public-default bitwise
   parity, empty/masked preflight, raw samples, stage measurements, source
   snapshots, process inventory, and environment. Keep all server artifacts and
   caches under `~/nhl_project`, using existing nhlpy3. Monitor other compute
   processes every five seconds and before/after cases. If interference occurs,
   stop only our child, retain the failure and repeat the entire affected
   session later; never remove unfavorable individual timings.

The two sessions are not two GPU architectures or a population of servers.
Visibility and periodic guards do not reserve the GPU. No host allocation,
policy computation, metadata, JIT, H2D, cold-cache or full-model time is included.

## Grouped selection and comparators

Use the version 1 CPU-only model functions without changing them: default,
constant S in `{1,4,8,9,16,32}`, affine, and physics nonnegative ridge, lambda
`{0.001,0.01,0.1,1.0}`, scaling, weights, local margin and tie breaks. The
fixed constant menu intentionally remains the v1 menu; S2 is still measured
and available to regression decisions and the production baseline.

Leave out one **whole shape family**, keeping all three profiles and all split
choices together, for four CV folds. Do not apply fold-specific support guards.
Eligibility requires at least 1% equal-case geometric improvement and no more
than 10% worst per-case slowdown. Unlike v1, any nonconverged fold makes that
model ineligible; nonconvergence in the final refit selects default. No optimizer
or threshold adjustment after held-out data. Refit the winning eligible family
on all A rows and freeze; default is always eligible.

Use the original marginal support guard, now fitted on the larger training
design. All 12 new held-out cases are inside these marginal bounds. This does
not guarantee coverage of their interactions. Retain the previous frozen
policy, SHA256
`005fed5ad030f2f3cb891b7bb0942a5257a068092b8decf4adaedd8544c4678d`,
as an unchanged comparator. Also report its predictions **without its guard**
as an offline diagnostic only. This distinguishes guard coverage from ranking
error; it does not justify disabling the production or research guard.

## Reporting and uncertainty

Report each session separately and pooled: case median, MAD/IQR, default and
selected S, absolute latency change, equal-case geometric speedup, practical
wins/losses beyond max(0.1us,1%), sampled-grid best and regret. Report profile
subgroups and matched skew/quarter ratios with equal tokens. Explain stage
contributions with measured producer and finish timings. Record all regressions
and the default-policy outcome if no model qualifies.

Cross-session selector: select the grid winner in B and evaluate it in C, then
reverse. The same-session minimum is descriptive and optimistically selected,
not an independent performance target. For an exploratory interval, use 2,000
bootstrap replicates with seed 20261026: resample two session clusters, four
family clusters with all their profiles kept together, then paired candidate
round indices within each case. There are only two sessions and four held-out
families; the interval is descriptive, not a robust population guarantee.
128 repeated graph calls are not 128 independent samples.

A capacity-only model may fail on matched-token skew even after support expands.
If grid headroom remains but the frozen model fails, report that negative result
and propose a new version using fresh held-outs. Do not tune this model on B/C,
reinterpret training cases as tests, or deploy a runtime policy from these
measurements alone.
