# RTX 6000D reduction optimization

The final change speeds up the reduction/finish stage and removes it for short
single-split requests. Across this 13-case matrix, full GPU forward has an
**unweighted geometric-mean speedup of 1.097x**, with observed speedups from
1.006x to 1.919x. The ten retained reduction/finish kernels have a geometric-mean
speedup of **2.06x**, ranging from 1.32x to 5.44x. Three cases eliminate
the separate kernel and are excluded from that reduction-speedup average.

All measured full-forward medians improved. The small improvements near 0.6%
should be treated cautiously on a shared machine with unlocked clocks. The
largest gain is the 64-request, 64-token case: **45.041 to 23.477 microseconds**
(1.919x). For the original one-request, 32768-token BF16 case, reduction improves
from **3.447 to 2.413 microseconds**, while full forward improves only from
**157.579 to 156.577 microseconds** because attention dominates that workload.

## Changes

The attention tile sizes, split-count heuristic, public API and numerical
precision are retained. The RTX finishing policy is now:

| Condition | Execution |
|---|---|
| One split, allocated table capacity at most 256 tokens | Attention writes the final FP16/BF16 output and natural-log LSE directly. No partial buffers or merge kernel. |
| One split, larger allocated capacity | Preserve the FP32 attention epilogue; a linear 1024-element, four-warp copy kernel converts two rows per block and changes the LSE layout/base. |
| 2–8 splits | Weighted merge with 256 value columns per block and two warps. |
| 9–32 splits | Weighted merge with 128 value columns per block and one warp. |

The previous merge used 128 columns and four warps for every split count. The
measured alternatives reduce block/warp overhead while retaining stable FP32
weighted reduction. There is no runtime autotuning, GPU-to-CPU length read, or
new synchronization in the forward path, so CUDA graph replay remains supported.
The direct-output decision uses allocated table capacity, not a changing live
length; its route therefore remains fixed during graph replay.

Direct output avoids `rows * (512 + 1) * 4` bytes of FP32 scratch. This removes
262,656 bytes in the one-request short cases and 16,809,984 bytes (16.03 MiB) in
the 64-request short case. The retained merge paths keep their previous scratch
size and FP32 partials.

The design follows measurement. Direct output for the large-batch long-context
case regressed full forward, so that case retains the original attention
epilogue. Grouping multiple output rows in a weighted-merge block was also
slower than the chosen launch policy. Those experimental kernels are not used
in the final implementation; their measurements and source snapshots remain
in the server audit directory.

## Before and after

Times are microseconds per call, lower is better. Shape is
`batch / query tokens / query heads / KV heads / cache tokens / dtype`.
All cases use QK dimension 576, value dimension 512, 64-token pages and causal
attention. Reduction and full forward are measured independently.

| Shape | Splits | Reduce before | Reduce after | Reduce speedup | Full before | Full after | Full speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1/1/128/1/64/bf16 | 1 | 1.076 | eliminated | — | 8.257 | 6.597 | 1.252x |
| 1/1/128/1/256/bf16 | 1 | 1.219 | eliminated | — | 22.370 | 20.469 | 1.093x |
| 1/1/128/1/1024/bf16 | 4 | 2.526 | 1.356 | 1.86x | 23.913 | 22.787 | 1.049x |
| 1/1/128/1/4096/bf16 | 16 | 2.873 | 1.934 | 1.48x | 27.008 | 25.898 | 1.043x |
| 1/1/128/1/32768/bf16 | 32 | 3.447 | 2.413 | 1.43x | 157.579 | 156.577 | 1.006x |
| 4/1/128/1/4096/bf16 | 10 | 5.742 | 2.859 | 2.01x | 91.393 | 88.672 | 1.031x |
| 16/1/128/1/4096/bf16 | 3 | 17.226 | 3.165 | 5.44x | 317.506 | 303.424 | 1.046x |
| 64/1/128/1/4096/bf16 | 1 | 17.808 | 4.154 | 4.29x | 1543.403 | 1532.661 | 1.007x |
| 64/1/128/1/64/bf16 | 1 | 17.808 | eliminated | — | 45.041 | 23.477 | 1.919x |
| 1/4/128/1/4096/bf16 | 10 | 5.590 | 2.730 | 2.05x | 93.517 | 90.863 | 1.029x |
| 1/1/16/1/32768/bf16 | 32 | 1.826 | 1.382 | 1.32x | 73.706 | 73.287 | 1.006x |
| 4/2/64/2/8192/bf16 | 10 | 5.717 | 2.842 | 2.01x | 174.304 | 171.715 | 1.015x |
| 1/1/128/1/32768/fp16 | 32 | 3.528 | 2.539 | 1.39x | 157.670 | 156.624 | 1.007x |

“Eliminated” means no separate reduction kernel is launched; it is not a
zero-duration measured kernel. The long-context single-split “after” measurement
is the copy/conversion kernel. Isolated stage medians need not sum exactly to
the full-forward median because cache state and measurement boundaries differ.

## Method and reproducibility

The baseline is the immutable RTX backend in commit
`1f168a52a4bce6d34439991ee63070d6a0c49825`. The benchmark loads that source with `git show`
into the selected output directory. Both versions run in the same process with
the same tensors, dtype, cache page permutation, scale and split count. This
avoids comparing against the earlier port's timing snapshot, which used a
different timing loop and GPU clock state.

Environment: NVIDIA RTX 6000D, 156 SMs, compute capability 12.0, driver 595.99.02;
Python 3.12.3 in `~/nhl_project/nhlpy3`; PyTorch 2.9.1+cu128 and Triton 3.5.1.
Physical GPU 0 was idle before each run. GPU clocks and power limits were not
changed. The final benchmark recorded GPU temperature rising to 56 degrees C
and SM clock 2422 MHz at its end; these are observations, not locked settings.

Each timing sample uses CUDA events around a CUDA graph containing **128 calls**.
This avoids Python replay gaps dominating microsecond reduction kernels.
Candidates are warmed and JIT-compiled before measurement. **15 rounds** rotate
and reverse the order of the paired candidates; the table reports the median.
The JSON includes every sample, minimum, maximum, source hashes and environment.
The buffers are resident and warm; no deliberate L2 flush is used. Full-forward
graphs include all GPU work for the forward, with captured allocations, and
exclude Python allocation overhead, metadata creation, JIT, and H2D transfers.
This is an operator microbenchmark, not an application throughput claim or a
comparison with H100.

From the repository, select an idle GPU after checking `nvidia-smi`:

```bash
cd ~/nhl_project/moore_tma_pj/kernels/FlashMLA_old_opt
source scripts/rtx6000d_env.sh
nvidia-smi
CUDA_VISIBLE_DEVICES=0 python tests/test_sm120.py
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_sm120_reduce.py \
  --output ~/nhl_project/moore_tma_pj/docs/reduce_recheck --rounds 15
```

Replace GPU index `0` with the free device. `--tune` repeats the merge launch
sweep; `--tune-copy` repeats the one-split conversion sweep. Output must be under
`~/nhl_project`. No conda, global install, clock locking or system configuration
change is needed. The existing project environment and caches are reused.

See [the benchmark source](../benchmarks/benchmark_sm120_reduce.py) and
[complete timing and correctness results](RTX6000D_REDUCE_RESULTS.json).
Final RTX source SHA256: `764f799a5ac19e9f48c35a1ba64efd7d05fb03d60cede78a47263838dabaedf1`.
Raw baseline, intermediate candidates, final logs and backups are under:

```text
~/nhl_project/moore_tma_pj/docs/rtx6000d_reduce_20261023/
```

## Correctness and preservation

All **seven test methods passed**, with no skips. They include the existing
20 BF16/FP16 numerical configurations, three CUDA-graph reference checks with
changing live lengths, input validation, and CPU/unsupported-device rejection.
New coverage includes **88 direct reduction comparisons against FP64 reference
weights**: two dtypes, four row shapes, and eleven split counts from 1 through
32, including uneven counts. These test extreme LSE differences, empty rows,
and row tails. An additional full-attention reference check confirms that the
short single-split path does not call the merge launcher. All 13 benchmark
cases also compare the candidate output and LSE against the baseline before
timing. No tolerances were loosened to obtain a pass.

All 11 tracked H100 CUDA source files retain their original hashes. The
Hopper build, dispatch, CUTLASS submodule and original kernel experiments are
unchanged. H100 runtime correctness/performance was not re-tested because this
server provides RTX GPUs. The retained H100 design is described in
[TMA_REDUCTION.md](TMA_REDUCTION.md).
