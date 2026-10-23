"""Controlled row x split sweep for the SM120 weighted reducer only.

This isolates split count from the attention scheduler. It does not measure
attention, TMA, model decoding, device-memory traffic or energy consumption.
Use scripts/rtx6000d_env.sh and an idle CUDA_VISIBLE_DEVICES device.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import torch
import triton

from benchmark_sm120_reduce import BASE, REPO, baseline_module, gpu_state, paired_times
from flash_mla_old import sm120 as current


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rounds', type=int, default=15)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds must be positive')
    output = Path(args.output).expanduser().resolve()
    if (Path.home() / 'nhl_project').resolve() not in output.parents:
        raise ValueError('--output must be under ~/nhl_project')
    output.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.get_device_capability() == (12, 0), 'SM120 required'
    torch.manual_seed(20261024)
    torch.set_num_threads(2)
    baseline, baseline_hash = baseline_module(output)
    result = {
        'suite': 'controlled-split-grid', 'seed': 20261024,
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'baseline_commit': BASE, 'baseline_sha256': baseline_hash,
        'candidate_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'tracked_diff': subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=REPO, text=True).splitlines(),
        'candidate_sha256': hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
        'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'timing_helper_sha256': hashlib.sha256((REPO / 'benchmarks/benchmark_sm120_reduce.py').read_bytes()).hexdigest(),
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'triton': triton.__version__, 'cuda': torch.version.cuda,
        'gpu_before': gpu_state(), 'rounds': args.rounds,
        'timing': 'CUDA events; 128 calls per graph; rotating paired rounds; warm resident synthetic FP32 partials; BF16 output; no clock changes',
        'reference': 'FP64 softmax of natural-log LSE and weighted sum, then BF16 output rounding',
        'cases': [],
    }
    for rows in (128, 256, 512, 1024):
        for splits in (1, 2, 4, 8, 9, 16, 24, 32):
            partial = torch.randn((rows, splits, 512), device='cuda', dtype=torch.float32)
            partial_lse = torch.randn((rows, splits), device='cuda', dtype=torch.float32) * 5
            partial[0] = 0
            partial_lse[0] = -float('inf')
            if splits > 1:
                partial_lse[-1, ::2] = -float('inf')
                partial[-1, ::2] = 0
            out = torch.empty((1, 1, rows, 512), device='cuda', dtype=torch.bfloat16)
            lse = torch.empty((1, rows, 1), device='cuda', dtype=torch.float32)

            def before():
                baseline._merge_splits[(rows, 4)](
                    partial, partial_lse, out, lse, 1, rows, splits,
                    triton.next_power_of_2(splits), 128, num_warps=4)

            def after():
                current._launch_merge(partial, partial_lse, out, lse, 1, rows, splits)

            natural = partial_lse.double() * math.log(2)
            weights = torch.nan_to_num(natural.softmax(-1), nan=0.0)
            expected = (partial.double() * weights[..., None]).sum(1).reshape_as(out).to(out.dtype)
            expected_lse = natural.logsumexp(-1).reshape_as(lse).float()
            checks = {}
            for name, fn in [('before', before), ('after', after)]:
                fn()
                torch.testing.assert_close(out, expected, atol=0.002, rtol=0.01)
                torch.testing.assert_close(lse, expected_lse, atol=0.001, rtol=2e-6)
                finite = torch.isfinite(expected_lse)
                checks[name] = {
                    'max_abs_output_error': (out.float() - expected.float()).abs().max().item(),
                    'max_abs_finite_lse_error': (lse[finite] - expected_lse[finite]).abs().max().item(),
                    'empty_row_zero_and_negative_infinity': bool((out[0, 0, 0] == 0).all() and torch.isneginf(lse[0, 0, 0])),
                }
                assert checks[name]['empty_row_zero_and_negative_infinity']
            timings = paired_times({'before': before, 'after': after}, args.rounds)
            record = {
                'rows': rows, 'splits': splits, 'dtype': 'bf16',
                'partial_bytes': rows * splits * 513 * 4,
                'logical_read_write_bytes': rows * splits * 513 * 4 + rows * (512 * 2 + 4),
                'reference_checks': checks, 'timings': timings,
                'speedup': timings['before']['median_us'] / timings['after']['median_us'],
            }
            result['cases'].append(record)
            (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps({'rows': rows, 'splits': splits, 'speedup': record['speedup']}), flush=True)
    result['gpu_after'] = gpu_state()
    result['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
