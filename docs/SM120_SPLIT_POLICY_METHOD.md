# Preregistered SM120 split-policy experiment

Protocol `sm120-split-policy-v1-20261025`; frozen before any new split-policy timings. Canonical protocol is `benchmarks/sm120_split_policy_protocol.json`, SHA256 `4fa96c41d85486a41b43a66efd1688cddaff31dc4ce83820c03275c1c30927d7`. Existing 24-case forward and 32-case synthetic reducer measurements were already inspected and are exploratory background only. New timing data, model coefficients, and evaluation outcomes are not yet known.

## Scope and primary question

On the current RTX 6000D with 156 SMs, can a policy trained on 12 workload cases choose split counts with better held-out **full producer plus finish** graph-replay latency than the current heuristic? The eight held-out cases, their candidate sets, fit algorithm, model features, eligibility thresholds, and summary rules are fixed in advance. This is an operator-level RTX experiment. It does not measure Hopper TMA, SM100 kernels, DeepInfer, model inference, policy CPU overhead, or serving latency.

The unchanged production heuristic is `min(32, ceil(capacity_tokens/256), max(1, ceil(312/(B*Hkv*ceil(Sq*(Hq/Hkv)/16)))))`. Candidate set is the sorted union of `{1,4,8,9,16,32,production_S}`; add 2 only if this union has fewer than seven elements. All twenty cases have exactly seven candidates. Fixed counts exceeding the live-length or occupancy heuristic create legal empty splits; correctness must still be checked independently. No GPU kernel, production dispatch, installation, clocks, or global settings change.

## Workload split

All cases use BF16, causal attention, page64, QK576, V512. Capacity is max(live lengths) rounded up to64. No test-case timing is used to choose inputs, candidates, model family, or hyperparameters.

| ID | Phase | B | Sq | Hq | Hkv | Live lengths |
|---|---|---:|---:|---:|---:|---|
| T01 | train | 1 | 1 | 16 | 1 | 128 |
| T02 | train | 1 | 1 | 128 | 1 | 1024 |
| T03 | train | 1 | 4 | 64 | 1 | 8192 |
| T04 | train | 2 | 1 | 64 | 1 | 4096,4096 |
| T05 | train | 2 | 2 | 128 | 2 | 32768,32768 |
| T06 | train | 4 | 1 | 128 | 1 | 256 each |
| T07 | train | 4 | 2 | 64 | 2 | 16384 each |
| T08 | train | 8 | 1 | 128 | 1 | 2048 each |
| T09 | train | 2 | 1 | 16 | 1 | 256,16384 |
| T10 | train | 4 | 4 | 64 | 2 | 1024,2048,4096,8192 |
| T11 | train | 4 | 2 | 128 | 2 | 64,512,4096,32768 |
| T12 | train | 8 | 1 | 64 | 2 | 256,256,1024,1024,4096,4096,16384,16384 |
| H01 | holdout | 1 | 2 | 16 | 1 | 3072 |
| H02 | holdout | 3 | 1 | 128 | 1 | 6144 each |
| H03 | holdout | 6 | 2 | 64 | 2 | 12288 each |
| H04 | holdout | 2 | 4 | 128 | 2 | 24576,24576 |
| H05 | holdout | 3 | 1 | 64 | 1 | 128,1536,12288 |
| H06 | holdout | 6 | 1 | 128 | 2 | 256,768,2048,4096,8192,24576 |
| H07 | holdout | 4 | 2 | 96 | 1 | 512,2048,8192,16384 |
| H08 | holdout | 8 | 4 | 128 | 2 | 128,128,512,512,2048,2048,8192,32768 |

H01-H07 test new combinations within the marginal training support. Hq96 is interpolation between seen head counts, not a performance claim for all head counts. H08 deliberately exceeds training total rows and producer groups and must select production default through the frozen support guard. Marginal bounds are a conservative operational guard, not proof that every interaction inside them is well trained.

## Measurement and independence

Use three fresh-process, newly acquired idle-GPU measurement sessions: A measures training cases only; B and C measure all twenty cases. B/C are independent process/initialization and shared-server acquisition intervals, not independent GPU architectures. Retain acquisition timestamps, GPU UUID/index, process/utilization checks, temperature/clocks, source hashes, software versions, manifest hash, input seed, and ordering. A continuous run merely split into two files is not two sessions. Do not reserve/reset another user's GPU or change device settings. If contention appears, stop and label the affected session invalid for timing; preserve the record and repeat the entire affected session at the next idle opportunity, not selectively unfavorable choices.

Each case and S requests at least15 timing rounds, with rounds rounded up to complete candidate-order cycles independently for each component group. The seven-choice full and producer groups therefore use21 rounds; finish uses21 when all seven choices have a finish, or18 when direct output leaves six finish choices. Each observation contains128 calls per CUDA graph, warm resident preallocated buffers, fixed kernel settings, and CUDA events. Full, producer, and finish are timed in separate groups. Candidate order rotates deterministically within each group and reverses on alternate complete cycles; it is not randomized. Preserve raw per-round samples and actual execution order. Full producer-plus-finish replay is timed directly; separately measured components are explanatory and are not added to make the full latency. Exclude JIT, setup, host-to-device copies, allocation, metadata and Python policy execution from the timed graph. The128 calls are repeated work within one observation, not128 independent measurements.

Independent FP32 reference checks and graph-replay correctness must pass for every candidate without relaxing existing tolerances. Preserve empty split, masked-row and LSE semantics. Failures are reported, not removed as slow cases. Correctness failure of a predeclared candidate invalidates that candidate's performance claim; pause the study for a versioned protocol amendment before model fitting rather than silently replacing choices. Sessions reuse the protocol-fixed per-case input seeds and reconstruct identical logical inputs in each fresh process. This isolates timing/session drift; it is not independent input-distribution sampling. Case lengths, candidate sets, and numerical tolerances remain unchanged.

## Leakage prevention and freeze

Only session A training rows enter fitting. The analyzer refuses any holdout timing rows in its freeze input. Leave-one-**whole-case**-out cross-validation keeps all S choices of the held-out training case together; no random row split mixes one shape's alternatives between training and validation.

After fitting A, write a new immutable JSON freeze record containing protocol/source/analyzer/training-file hashes, CV scores for all candidate families, coefficients/scales, support bounds, and split predictions for all20 cases. Record its SHA256 and creation time before B or C holdout timing is displayed or consumed by analysis. B/C benchmark runs require this inline policy record and its SHA256. Later evaluation verifies complete sessions/cases, passing preflight/reference/graph checks, unchanged source bundles,156SM environment, matching frozen-policy hashes, test starts after freeze, and distinct fresh-process session records. It then uses the frozen decisions with no refit. Training rows in B/C measure session transfer only; they do not update the model.

## Predeclared candidate models

Primary policies use only **host-visible dimensions and allocated capacity**, compatible in principle with graph capture and no CUDA-length host synchronization. The model does not read actual CUDA live lengths to make a choice. Heterogeneous live lengths affect measured runtime and the generalization test; they are not free deployment features. No runtime policy is installed in this experiment.

Let R=B*Sq*Hq, T=ceil(Sq*(Hq/Hkv)/16), P=B*Hkv*T, N=P*S, and K=ceil(capacity/32). Producer capacity work is `P*K/156`, launch waves `ceil(N/156)`, and longest-wave proxy `ceil(N/156)*ceil(K/S)`. These are capacity-based explanatory proxies; they do not claim actual occupancy, scheduling order, DRAM bytes, or measured SM waves. Direct output occurs only when S=1 and capacity<=256. Otherwise scratch is `4*R*S*513` bytes. Finish uses copy when S=1, two dimension tiles per row when2<=S<=8, and four dimension tiles whenS>=9. Padded reduction extent is `next_power_of_two(S)`.

Four families are defined before fitting:

1. **Default**: exact current heuristic.
2. **Constant**: one S from `{1,4,8,9,16,32}`, selected using training geometric mean latency relative to default. These choices exist for every case; this is a simple policy baseline, not per-case tuning.
3. **Affine**: nonnegative cost regression with intercept, producer capacity work, scratch MiB and finish-launch indicator.
4. **Physics**: the affine terms plus longest-wave proxy, producer waves, finish-grid waves, padded reduction work `R*nextpow2(S)*512/2^20` for S>1, and S>=9 dispatch indicator. Finish waves are ceil(copy_CTAs/156) for non-direct S1 or ceil(R*dimension_tiles/156) for S>1; zero for direct.

Fit CPU-only nonnegative ridge regression. Non-intercept features are scaled by their training RMS; response by training median latency. Weighted squared error gives each case weight proportional to inverse squared default latency, with per-choice weight divided by that case's candidate count. This prevents long cases dominating solely through units. Intercept is not penalized. Lambda candidates are exactly `{0.001,0.01,0.1,1.0}`. Coordinate descent begins at zero and stops at maximum coefficient change<1e-9 or5000 sweeps. Nonconvergence must be retained in diagnostics; it cannot justify choosing a different optimizer after seeing holdout.

For affine/physics, choose the lowest predicted-cost candidate, breaking ties toward smallerS. Switch away from default only when predicted saving reaches `max(0.1 microsecond,1% of predicted default latency)`. For each family/lambda, evaluate the frozen local decision rule on12 leave-one-case-out folds. A non-default model is eligible only if CV geometric mean speedup>=1.01 and worst CV slowdown<=1.10. Select eligible family/lambda with lowest mean log latency ratio to default; ties favor default, constant, affine, physics, then smaller lambda. Default is always eligible. Refit only the selected family on all A training cases and freeze. The constant family's gain gate is global CV; it has no predicted-time local margin.

Support guard uses training minima/maxima of B,Sq,Hq,Hkv,query-heads-per-KV-head,R,T,P,and capacity, plus BF16 causal contract. Any out-of-range value selects default. During CV, do not apply these fold-specific bounds, because leaving an extreme case out would change the intended final support domain and hide modeling errors; final support bounds come from the fixed12-case training design only. H08 is therefore a declared fallback test, not a learned-policy success.

## Outcomes and uncertainty

Primary results: per-heldout-case medians, selected/default splits, full-pipeline speedup, absolute microsecond difference, and equal-case geometric mean speedup for B and C separately and pooled. Report H01-H07 (learned support) and all8 including fallback separately. Declare a per-case practical win/loss only beyond `max(0.1us,1% of default median)`; otherwise practical tie. Do not convert many within-graph calls into a large sample count.

Report sampled candidate-grid regret `(selected_latency/min_candidate_latency)-1`, absolute regret, and default's oracle headroom. This minimum is noisy and optimistic. Therefore additionally choose the measured best S using sessionB and evaluate that selector on C, then reverse B/C. The frozen policy's gap against this cross-session selector can be negative and must not be clipped. This separates policy error from noisy winner selection. The grid optimum is only the best of the seven measured choices, not a global optimum over all split counts.

Retain raw paired round arrays, medians, MAD/IQR, min/max and session differences. A fixed-seed2000-replicate paired hierarchical bootstrap may resample test sessions, then case IDs, then paired round indices; report it as an exploratory interval with only two session clusters, not a robust population confidence guarantee. Within-case timing observations are correlated and the workload set is purposive. Report session-specific signs and magnitudes rather than claiming significance from tiny differences.

Useful negative outcomes are predetermined: (a) default has little repeatable candidate-grid headroom, so this domain does not justify extra policy complexity; (b) grid headroom exists but the model does not generalize, so capacity/wave/scratch features are insufficient; (c) apparent gains disappear across sessions, so noise or shared-server state dominates; (d) uniform cases improve but heterogeneous cases fail, exposing capacity-only work estimation; (e) the guard selects default, an explicit conservative result. Distinguish these outcomes. No new model, threshold, candidate, or case can be adopted using B/C results without a new version and fresh untouched holdout.

