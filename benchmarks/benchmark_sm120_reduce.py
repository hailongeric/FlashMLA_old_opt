"""Paired, graph-batched GPU timings against the immutable initial RTX backend.

Run after sourcing scripts/rtx6000d_env.sh; select an idle GPU externally with
CUDA_VISIBLE_DEVICES. --output must be under ~/nhl_project on the shared server.
No clocks, power limits or other system settings are modified.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
import triton

from flash_mla_old import sm120 as current

BASE = '1f168a52a4bce6d34439991ee63070d6a0c49825'
REPO = Path(__file__).resolve().parents[1]
CASES = [
    (1, 1, 128, 1, 64, 'bf16'),
    (1, 1, 128, 1, 256, 'bf16'),
    (1, 1, 128, 1, 1024, 'bf16'),
    (1, 1, 128, 1, 4096, 'bf16'),
    (1, 1, 128, 1, 32768, 'bf16'),
    (4, 1, 128, 1, 4096, 'bf16'),
    (16, 1, 128, 1, 4096, 'bf16'),
    (64, 1, 128, 1, 4096, 'bf16'),
    (64, 1, 128, 1, 64, 'bf16'),
    (1, 4, 128, 1, 4096, 'bf16'),
    (1, 1, 16, 1, 32768, 'bf16'),
    (4, 2, 64, 2, 8192, 'bf16'),
    (1, 1, 128, 1, 32768, 'fp16'),
]
LOW_BATCH_CASES = [(b, 1, 128, 1, length, 'bf16')
                   for b in (1, 2, 4, 8)
                   for length in (64, 256, 1024, 4096, 16384, 32768)]
CONFIGS = [(64, 2), (64, 4), (128, 1), (128, 2), (128, 4),
           (256, 1), (256, 2), (256, 4), (512, 2), (512, 4), (512, 8)]


def baseline_module(output):
    src = subprocess.check_output(['git', 'show', f'{BASE}:flash_mla_old/sm120.py'], cwd=REPO)
    path = output / 'sm120_baseline.py'
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location('_flashmla_sm120_baseline', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, hashlib.sha256(src).hexdigest()


def graph_for(fn, repetitions):
    # Warm/JIT on a side stream, then capture many calls in one graph to remove
    # Python replay gaps from microsecond-scale kernel measurements.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repetitions):
            fn()
    return graph


def paired_times(functions, rounds, repetitions=128):
    graphs = {name: graph_for(fn, repetitions) for name, fn in functions.items()}
    samples = {name: [] for name in graphs}
    names = list(graphs)
    for graph in graphs.values():
        for _ in range(3):
            graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for round_id in range(rounds):
        order = names[round_id % len(names):] + names[:round_id % len(names)]
        if round_id % 2:
            order = list(reversed(order))
        for name in order:
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / repetitions)
    return {name: {'median_us': statistics.median(values), 'min_us': min(values),
                   'max_us': max(values), 'samples_us': values} for name, values in samples.items()}


def setup(case, baseline):
    b, sq, hq, hk, length, dtype_name = case
    dtype = torch.bfloat16 if dtype_name == 'bf16' else torch.float16
    q = torch.randn((b, sq, hq, 576), dtype=dtype, device='cuda')
    cache = torch.randn((b * triton.cdiv(length, 64), 64, hk, 576), dtype=dtype, device='cuda')
    table = torch.randperm(cache.shape[0], device='cuda').to(torch.int32).reshape(b, -1)
    lengths = torch.full((b,), length, dtype=torch.int32, device='cuda')
    meta = baseline.get_mla_metadata(lengths, sq * hq // hk, hk)
    args = (q, cache, table, lengths, 512, *meta, 576 ** -0.5, True)
    rows, row_tiles = b * sq * hq, triton.cdiv(sq * (hq // hk), 16)
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    splits = min(32, triton.cdiv(table.shape[1] * 64, 256), max(1, triton.cdiv(2 * sm_count, b * hk * row_tiles)))
    partial = torch.empty((rows, splits, 512), dtype=torch.float32, device='cuda')
    partial_lse = torch.empty((rows, splits), dtype=torch.float32, device='cuda')
    out = torch.empty((b, sq, hq, 512), dtype=dtype, device='cuda')
    lse = torch.empty((b, hq, sq), dtype=torch.float32, device='cuda')
    def attention():
        baseline._split_attention[(row_tiles, splits, b * hk)](
            q, cache, table, lengths, partial, partial_lse,
            sq, hq, hk, *q.stride()[:3], *cache.stride()[:3],
            table.stride(0), table.shape[1], cache.shape[0], splits,
            576 ** -0.5 * 1.4426950408889634, True, 16, 32,
            num_warps=4, num_stages=1)
    def merge(bd=128, warps=4):
        baseline._merge_splits[(rows, 512 // bd)](
            partial, partial_lse, out, lse, sq, hq, splits,
            triton.next_power_of_2(splits), bd, num_warps=warps)
    attention()
    merge()
    return args, splits, (partial, partial_lse, out, lse), attention, merge


def gpu_state():
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    return subprocess.check_output(['nvidia-smi', '-i', physical, '--query-gpu=index,name,driver_version,utilization.gpu,memory.used,clocks.sm,clocks.mem,temperature.gpu,power.draw', '--format=csv,noheader'], text=True).strip()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--tune-copy', action='store_true')
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--suite', choices=('default', 'low-batch'), default='default')
    parser.add_argument('--check-reference', action='store_true',
                        help='Check both backends against the independent FP32 attention reference before timing')
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds must be positive')
    cases = CASES if args.suite == 'default' else LOW_BATCH_CASES
    output = Path(args.output).expanduser().resolve()
    project = Path.home() / 'nhl_project'
    if project.resolve() not in output.parents:
        raise ValueError('--output must be under ~/nhl_project')
    output.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.get_device_capability() == (12, 0)
    torch.manual_seed(20261023)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    baseline, baseline_hash = baseline_module(output)
    reference = None
    if args.check_reference:
        spec = importlib.util.spec_from_file_location('_flashmla_reference', REPO / 'tests/test_sm120.py')
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
    props = torch.cuda.get_device_properties(0)
    result = {'suite': args.suite, 'seed': 20261023,
              'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'candidate_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
              'tracked_diff': subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=REPO, text=True).splitlines(),
              'reference_checked': args.check_reference,
              'reference_sha256': hashlib.sha256((REPO / 'tests/test_sm120.py').read_bytes()).hexdigest() if reference else None,
              'baseline_commit': BASE, 'baseline_sha256': baseline_hash,
              'candidate_sha256': hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
              'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'gpu': props.name, 'sm_count': props.multi_processor_count,
              'torch': torch.__version__, 'triton': triton.__version__,
              'cuda': torch.version.cuda, 'rounds': args.rounds,
              'timing': 'CUDA events; 128 calls per CUDA graph; alternating/rotating paired rounds; warm resident buffers; no clock changes; excludes JIT, metadata and H2D',
              'gpu_before': gpu_state(), 'cases': []}
    print(json.dumps({k:v for k,v in result.items() if k != 'cases'}), flush=True)
    # Bring the GPU out of its idle clock state without altering global settings.
    warm_args, _, _, _, _ = setup(CASES[4], baseline)
    warm = graph_for(lambda: baseline.flash_mla_with_kvcache(*warm_args), 128)
    for _ in range(8):
        warm.replay()
    torch.cuda.synchronize()
    del warm, warm_args
    for case in cases:
        inputs, splits, buffers, attention, merge = setup(case, baseline)
        q = inputs[0]
        expected = baseline.flash_mla_with_kvcache(*inputs)
        actual = current.flash_mla_with_kvcache(*inputs)
        torch.testing.assert_close(actual[0], expected[0], atol=0.001, rtol=0.01)
        torch.testing.assert_close(actual[1], expected[1], atol=2e-4, rtol=2e-5)
        checks = {}
        if reference:
            ref_out, ref_lse = reference.reference(*inputs[:4], scale=inputs[-2], causal=inputs[-1])
            for name, (out, lse) in [('before', expected), ('after', actual)]:
                reference.check((*inputs[:4], None), causal=inputs[-1], scale=inputs[-2], result=(out, lse))
                checks[name] = {'max_abs_output_error': (out.float() - ref_out).abs().max().item(),
                                'max_abs_lse_error': (lse - ref_lse).abs().max().item(),
                                'cosine_error': reference.METRICS[-1]['cosine_error']}
            del ref_out, ref_lse
        direct = current._uses_direct_output(splits, inputs[2].shape[1]) if hasattr(current, '_uses_direct_output') else False
        def attention_after():
            b, sq, hq, hk, length, _ = case
            cache, table, lengths = inputs[1:4]
            row_tiles = triton.cdiv(sq * (hq // hk), 16)
            dest, dest_lse = (buffers[2], buffers[3]) if direct else buffers[:2]
            current._split_attention[(row_tiles, splits, b * hk)](
                q, cache, table, lengths, dest, dest_lse,
                sq, hq, hk, *q.stride()[:3], *cache.stride()[:3],
                table.stride(0), table.shape[1], cache.shape[0], splits,
                576 ** -0.5 * 1.4426950408889634, True, 16, 32,
                DIRECT=direct, num_warps=4, num_stages=1)
        timings = paired_times({'attention_before': attention,
                               'attention_after': attention_after,
                               'reduce_before': merge,
                               'full_before': lambda: baseline.flash_mla_with_kvcache(*inputs),
                               'full_after': lambda: current.flash_mla_with_kvcache(*inputs)}, args.rounds)
        record = {'reference_checks': checks, 'shape': list(case), 'splits': splits, 'rows': math.prod(q.shape[:3]),
                  'scratch_before_bytes': buffers[0].numel() * 4 + buffers[1].numel() * 4,
                  'direct_output': direct, 'scratch_after_bytes': 0 if direct else buffers[0].numel() * 4 + buffers[1].numel() * 4,
                  'timings': timings}
        record['full_speedup'] = timings['full_before']['median_us'] / timings['full_after']['median_us']
        if hasattr(current, '_launch_merge') and not direct:
            record['merge_comparison'] = paired_times({
                'before': merge,
                'after': lambda: current._launch_merge(*buffers, q.shape[1], q.shape[2], splits)
            }, args.rounds)
            record['reduce_speedup'] = record['merge_comparison']['before']['median_us'] / record['merge_comparison']['after']['median_us']
        if args.tune:
            variants = {f'bd{bd}_w{warps}': (lambda bd=bd, warps=warps: merge(bd, warps)) for bd, warps in CONFIGS}
            for fn in variants.values():
                fn()
                torch.testing.assert_close(buffers[2], expected[0], atol=0.001, rtol=0.01)
                torch.testing.assert_close(buffers[3], expected[1], atol=2e-4, rtol=2e-5)
            record['tuning'] = paired_times(variants, args.rounds)
        if args.tune_copy and splits == 1:
            variants = {'selected': lambda: current._launch_merge(*buffers, q.shape[1], q.shape[2], splits)}
            for block in (512, 1024, 2048, 4096):
                for warps in (4, 8):
                    def launch(block=block, warps=warps):
                        current._copy_single[(triton.cdiv(record['rows'] * 512, block),)](
                            *buffers, record['rows'], q.shape[1], q.shape[2], block, num_warps=warps)
                    launch()
                    torch.testing.assert_close(buffers[2], expected[0], atol=0.001, rtol=0.01)
                    torch.testing.assert_close(buffers[3], expected[1], atol=2e-4, rtol=2e-5)
                    variants[f'block{block}_w{warps}'] = launch
            record['copy_tuning'] = paired_times(variants, args.rounds)
        result['cases'].append(record)
        (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        summary = {'case': case, 'splits': splits,
                   'times_us': {k:round(v['median_us'], 3) for k,v in timings.items()}}
        if args.tune:
            best = min(record['tuning'], key=lambda key:record['tuning'][key]['median_us'])
            summary['best'] = [best, round(record['tuning'][best]['median_us'], 3)]
        print(json.dumps(summary), flush=True)
    result['gpu_after'] = gpu_state()
    result['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
