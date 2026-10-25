"""Protocol-driven fixed-split research, without changing production defaults.

Source scripts/rtx6000d_env.sh and select one idle CUDA_VISIBLE_DEVICES GPU.
Protocol schema_version=1: protocol_id, seed, cases. Each case supplies id,
phase (train/holdout), batch, sq, hq, hk, dtype (bf16/fp16), allocated_length
(positive multiple of64), live_lengths, candidate_splits (distinct1..32 and
including production's choice), causal, optional seed/softmax_scale.

Train is the default phase. Holdout/all also require --freeze-record and its
externally recorded --freeze-record-sha256. The JSON record binds protocol_sha256,
an inline model and predictions, or policy_artifact (relative to the record or
absolute) and policy_sha256.
--validate-protocol-only needs no GPU dependencies and creates no output.

Full timings are preallocated producer plus current finish route in CUDA Graphs.
Allocations, metadata, JIT, H2D, reference and framework/model work are excluded.
Finish-only repeatedly reads resident partials; direct output has no finish
kernel and therefore no finish timing. This program neither fits nor deploys
a split policy. All source snapshots, raw timings and ordering are retained.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from unittest import mock

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
TOLERANCES = {'bf16': {'atol': 0.006, 'rtol': 0.02},
              'fp16': {'atol': 0.001, 'rtol': 0.005},
              'lse': {'atol': 2e-4, 'rtol': 2e-5}, 'cosine_error_limit': 1e-5}


def utc_now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def positive_int(value):
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('must be a positive integer')
    if result < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def integer(value, label, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{label} must be an integer in [{minimum}, {maximum or "unbounded"}]')


def hash_string(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value.lower()):
        raise ValueError(f'{label} must be a SHA256 hex digest')
    return value.lower()


def read_protocol(path):
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if not isinstance(protocol, dict) or protocol.get('schema_version') != 1:
        raise ValueError('protocol needs schema_version=1')
    if not isinstance(protocol.get('protocol_id'), str) or not protocol['protocol_id']:
        raise ValueError('protocol_id must be a nonempty string')
    integer(protocol.get('seed'), 'protocol seed', 0, 2**63 - 1)
    if not isinstance(protocol.get('cases'), list) or not protocol['cases']:
        raise ValueError('cases must be a nonempty list')
    seen = set()
    for index, case in enumerate(protocol['cases']):
        if not isinstance(case, dict):
            raise ValueError(f'case {index} must be an object')
        cid = case.get('id')
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError('case ids must be distinct nonempty strings')
        seen.add(cid)
        if case.get('phase') not in ('train', 'holdout'):
            raise ValueError(f'{cid}: phase must be train or holdout')
        for key in ('batch', 'sq', 'hq', 'hk', 'allocated_length'):
            integer(case.get(key), f'{cid}.{key}')
        if case['allocated_length'] % 64 or case['hq'] % case['hk']:
            raise ValueError(f'{cid}: allocation must be a multiple of64; KV heads must divide query heads')
        if case.get('dtype') not in ('bf16', 'fp16') or type(case.get('causal')) is not bool:
            raise ValueError(f'{cid}: dtype must be bf16/fp16 and causal must be boolean')
        lengths = case.get('live_lengths')
        if not isinstance(lengths, list) or len(lengths) != case['batch']:
            raise ValueError(f'{cid}: live_lengths must contain batch values')
        for length in lengths:
            integer(length, f'{cid}.live_lengths', 0, case['allocated_length'])
        splits = case.get('candidate_splits')
        if not isinstance(splits, list) or not splits:
            raise ValueError(f'{cid}: candidate_splits must be a nonempty list')
        for split in splits:
            integer(split, f'{cid}.candidate_splits', 1, 32)
        if len(set(splits)) != len(splits):
            raise ValueError(f'{cid}: duplicate split choice')
        integer(case.get('seed', protocol['seed'] + index), f'{cid}.seed', 0, 2**63 - 1)
        scale = case.get('softmax_scale', 576 ** -0.5)
        if type(scale) not in (int, float) or not math.isfinite(scale):
            raise ValueError(f'{cid}: softmax_scale must be finite')
    return protocol, raw


def validate_freeze(args, protocol_hash):
    if args.phase in ('holdout', 'all') and (not args.freeze_record or not args.freeze_record_sha256):
        raise ValueError('holdout/all requires --freeze-record and --freeze-record-sha256 before GPU work')
    if not args.freeze_record and not args.freeze_record_sha256:
        return None
    if not args.freeze_record or not args.freeze_record_sha256:
        raise ValueError('provide both freeze-record arguments together')
    path = Path(args.freeze_record).expanduser().resolve(strict=True)
    raw = path.read_bytes()
    record_hash = hash_string(args.freeze_record_sha256, 'freeze record hash')
    if sha256(raw) != record_hash:
        raise ValueError('freeze record SHA256 mismatch')
    record = json.loads(raw)
    if not isinstance(record, dict) or record.get('protocol_sha256') != protocol_hash:
        raise ValueError('freeze record must bind the exact protocol SHA256')
    value = record.get('policy_artifact')
    if value is None:
        if not isinstance(record.get('model'), dict) or not isinstance(record.get('predictions'), list) or not record['predictions']:
            raise ValueError('freeze record needs an inline model/predictions or a separately hashed policy artifact')
        return {'record_path': str(path), 'record_sha256': record_hash,
                'policy_path': str(path), 'policy_sha256': record_hash,
                'policy_representation': 'inline_record', 'record': record,
                '_record_bytes': raw, '_policy_bytes': raw}
    if not isinstance(value, str) or not value:
        raise ValueError('freeze record needs policy_artifact')
    policy = Path(value).expanduser()
    policy = (policy if policy.is_absolute() else path.parent / policy).resolve(strict=True)
    policy_raw = policy.read_bytes()
    policy_hash = hash_string(record.get('policy_sha256'), 'policy artifact hash')
    if sha256(policy_raw) != policy_hash:
        raise ValueError('frozen policy artifact SHA256 mismatch')
    return {'record_path': str(path), 'record_sha256': record_hash,
            'policy_path': str(policy), 'policy_sha256': policy_hash,
            'policy_representation': 'separate_artifact', 'record': record,
            '_record_bytes': raw, '_policy_bytes': policy_raw}


def run_command(args, cwd=None, required=False):
    try:
        done = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=20)
        result = {'command': args, 'returncode': done.returncode,
                  'stdout': done.stdout.strip(), 'stderr': done.stderr.strip()}
    except (OSError, subprocess.SubprocessError) as exc:
        result = {'command': args, 'error': str(exc)}
    if required and result.get('returncode') != 0:
        raise RuntimeError(f'Cannot obtain required GPU state: {result}')
    return result


def gpu_state(selector):
    query = run_command(['nvidia-smi', '-i', selector,
        '--query-gpu=uuid,index,name,driver_version,pstate,utilization.gpu,memory.used,memory.total,clocks.sm,clocks.mem,temperature.gpu,power.draw',
        '--format=csv,noheader,nounits'], required=True)
    rows = list(csv.reader(io.StringIO(query['stdout']), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 12:
        raise RuntimeError('Expected exactly one identifiable GPU')
    columns = ('uuid', 'index', 'name', 'driver_version', 'pstate', 'utilization_gpu_percent',
               'memory_used_mib', 'memory_total_mib', 'clock_sm_mhz', 'clock_mem_mhz', 'temperature_c', 'power_draw_w')
    gpu = dict(zip(columns, rows[0]))
    processes = run_command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                             '--format=csv,noheader,nounits'], required=True)
    active = []
    for row in csv.reader(io.StringIO(processes['stdout']), skipinitialspace=True):
        if row and row[0] == gpu['uuid']:
            if len(row) != 4 or not row[1].isdigit():
                raise RuntimeError('Cannot interpret compute process inventory')
            active.append({'gpu_uuid': row[0], 'pid': int(row[1]), 'process_name': row[2], 'memory_mib': row[3]})
    return {'observed_utc': utc_now(), 'gpu': gpu, 'compute_processes': active,
            'foreign_compute_processes': [row for row in active if row['pid'] != os.getpid()],
            'raw_gpu_query': query, 'raw_compute_query': processes}


def require_unshared(state):
    if state['foreign_compute_processes']:
        raise RuntimeError('Another compute process occupies the selected GPU; deferred without changing or terminating it')


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save_result(output, result):
    temporary = output / 'results.json.tmp'
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(output / 'results.json')


def make_inputs(torch, case, seed):
    b, sq, hq, hk = (case[key] for key in ('batch', 'sq', 'hq', 'hk'))
    max_pages = case['allocated_length'] // 64
    dtype = torch.bfloat16 if case['dtype'] == 'bf16' else torch.float16
    generator = torch.Generator(device='cuda:0').manual_seed(seed)
    q = torch.randn((b, sq, hq, 576), generator=generator, dtype=dtype, device='cuda:0')
    cache = torch.randn((b * max_pages, 64, hk, 576), generator=generator, dtype=dtype, device='cuda:0')
    table = torch.randperm(cache.shape[0], generator=generator, device='cuda:0').to(torch.int32).reshape(b, max_pages)
    for batch, length in enumerate(case['live_lengths']):
        if length % 64:
            cache[table[batch, length // 64], length % 64:] = float('nan')
        table[batch, (length + 63) // 64:] = -1
    lengths = torch.tensor(case['live_lengths'], device='cuda:0', dtype=torch.int32)
    return q, cache, table, lengths


def make_pipeline(torch, triton, current, inputs, case, splits):
    q, cache, table, lengths = inputs
    b, sq, hq, hk = (case[key] for key in ('batch', 'sq', 'hq', 'hk'))
    rows, row_tiles = b * sq * hq, triton.cdiv(sq * (hq // hk), 16)
    direct = current._uses_direct_output(splits, table.shape[1])
    out = torch.empty((b, sq, hq, 512), device=q.device, dtype=q.dtype)
    lse = torch.empty((b, hq, sq), device=q.device, dtype=torch.float32)
    if direct:
        partial, partial_lse = out, lse
    else:
        partial = torch.empty((rows, splits, 512), device=q.device, dtype=torch.float32)
        partial_lse = torch.empty((rows, splits), device=q.device, dtype=torch.float32)

    def producer():
        current._split_attention[(row_tiles, splits, b * hk)](
            q, cache, table, lengths, partial, partial_lse,
            sq, hq, hk, *q.stride()[:3], *cache.stride()[:3],
            table.stride(0), table.shape[1], cache.shape[0], splits,
            case.get('softmax_scale', 576 ** -0.5) * 1.4426950408889634,
            case['causal'], 16, 32, DIRECT=direct, num_warps=4, num_stages=1)

    def finish():
        if not direct:
            current._launch_merge(partial, partial_lse, out, lse, sq, hq, splits)

    def full():
        producer()
        finish()

    route = 'direct' if direct else 'copy' if splits == 1 else 'merge'
    if direct:
        finish_ctas, config = 0, None
    elif splits == 1:
        finish_ctas, config = triton.cdiv(rows * 512, 1024), {'block': 1024, 'num_warps': 4}
    else:
        bd, warps = (256, 2) if splits <= 8 else (128, 1)
        finish_ctas = rows * (512 // bd)
        config = {'block_dim': bd, 'split_padding': triton.next_power_of_2(splits), 'num_warps': warps}
    return {'splits': splits, 'route': route, 'out': out, 'lse': lse,
            'partial': partial, 'partial_lse': partial_lse, 'producer': producer, 'finish': finish, 'full': full,
            'launch': {'producer_grid': [row_tiles, splits, b * hk], 'producer_ctas': row_tiles * splits * b * hk,
                       'producer_bm': 16, 'producer_bn': 32, 'producer_num_warps': 4, 'producer_num_stages': 1,
                       'finish_ctas': finish_ctas, 'finish_config': config,
                       'scratch_bytes': 0 if direct else rows * splits * 513 * 4,
                       'output_bytes': out.numel() * out.element_size() + lse.numel() * lse.element_size()}}


def check_result(torch, outputs, expected, dtype_name):
    out, lse = outputs
    ref_out, ref_lse = expected
    dtype = torch.bfloat16 if dtype_name == 'bf16' else torch.float16
    if out.dtype != dtype or out.shape != ref_out.shape or lse.dtype != torch.float32 or lse.shape != ref_lse.shape:
        raise AssertionError('Output shape/dtype contract mismatch')
    torch.testing.assert_close(out.float(), ref_out, **TOLERANCES[dtype_name])
    torch.testing.assert_close(lse, ref_lse, **TOLERANCES['lse'])
    cosine = ((out.float() - ref_out).square().sum() /
              (out.float().square().sum() + ref_out.square().sum()).clamp_min(1e-12)).item()
    if not math.isfinite(cosine) or cosine >= TOLERANCES['cosine_error_limit']:
        raise AssertionError(f'Cosine error {cosine} violates existing tolerance')
    finite, empty = torch.isfinite(ref_lse), torch.isneginf(ref_lse)
    empty_ok = bool(torch.all(out[empty.transpose(1, 2)] == 0).item() and torch.all(torch.isneginf(lse[empty])).item())
    if not empty_ok:
        raise AssertionError('Empty/fully masked rows must be exactly zero with negative-infinite LSE')
    return {'passed': True, 'max_abs_output_error': (out.float() - ref_out).abs().max().item(),
            'max_abs_finite_lse_error': (lse[finite] - ref_lse[finite]).abs().max().item() if finite.any().item() else 0.0,
            'cosine_error': cosine, 'empty_rows': int(empty.sum().item()),
            'empty_row_zero_and_negative_infinity': empty_ok,
            'output_finite': bool(torch.isfinite(out).all().item()), 'lse_nan_count': int(torch.isnan(lse).sum().item())}


def observe_public(torch, current, public_forward, public_metadata, inputs, case):
    q, cache, table, lengths = inputs
    calls, real_attention = [], current._split_attention

    class ForwardingRecorder:
        def __getitem__(self, grid):
            kernel = real_attention[grid]

            def launch(*args, **kwargs):
                calls.append({'grid': list(grid), 'splits_argument': args[18], 'direct': kwargs.get('DIRECT', False),
                              'num_warps': kwargs.get('num_warps'), 'num_stages': kwargs.get('num_stages')})
                return kernel(*args, **kwargs)
            return launch

    meta = public_metadata(lengths, case['sq'] * case['hq'] // case['hk'], case['hk'])
    with mock.patch.object(current, '_split_attention', ForwardingRecorder()), \
            mock.patch.object(current, '_launch_merge', wraps=current._launch_merge) as spy:
        outputs = public_forward(q, cache, table, lengths, 512, *meta,
                                 softmax_scale=case.get('softmax_scale', 576 ** -0.5), causal=case['causal'])
    torch.cuda.synchronize()
    if len(calls) != 1 or calls[0]['grid'][1] != calls[0]['splits_argument']:
        raise AssertionError('Unexpected public API producer launch')
    calls[0]['finish_launch_calls'] = spy.call_count
    return outputs, calls[0]


def balanced_order(count, round_index):
    offset = round_index % count
    order = list(range(offset, count)) + list(range(offset))
    if (round_index // count) % 2:
        order.reverse()
    return order


def paired_graph_times(torch, graph_for, pipelines, rounds, repetitions, record):
    # Full forward is the primary endpoint; stage timings are separate groups.
    functions = [(item['splits'], component, item[component])
                 for component in ('full', 'producer', 'finish') for item in pipelines
                 if component != 'finish' or item['route'] != 'direct']
    graphs = [graph_for(fn, repetitions) for _, _, fn in functions]
    for graph in graphs:
        for _ in range(3):
            graph.replay()
    torch.cuda.synchronize()
    samples = {(split, component): [] for split, component, _ in functions}
    record['measurement_order'] = observations = []
    record['timed_function_order'] = [{'splits': s, 'component': c} for s, c, _ in functions]
    record['measurement_groups'] = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for component in ('full', 'producer', 'finish'):
        indices = [index for index, (_, label, _) in enumerate(functions) if label == component]
        if not indices:
            continue
        effective_rounds = ((rounds + len(indices) - 1) // len(indices)) * len(indices)
        record['measurement_groups'].append({'component': component, 'requested_rounds': rounds,
            'effective_rounds': effective_rounds, 'candidate_order': [functions[index][0] for index in indices]})
        # Rewarm only this group's graphs, then measure its balanced paired rounds.
        for index in indices:
            graphs[index].replay()
        torch.cuda.synchronize()
        for round_index in range(effective_rounds):
            for position, local_index in enumerate(balanced_order(len(indices), round_index)):
                index = indices[local_index]
                split = functions[index][0]
                start.record()
                graphs[index].replay()
                end.record()
                end.synchronize()
                elapsed = start.elapsed_time(end) * 1000 / repetitions
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise RuntimeError('Invalid CUDA event timing')
                samples[(split, component)].append(elapsed)
                observations.append({'round': round_index, 'position': position, 'function_index': index,
                                     'splits': split, 'component': component, 'elapsed_us': elapsed})
    for index, (_, component, _) in enumerate(functions):
        if component == 'full':
            graphs[index].replay()
    torch.cuda.synchronize()
    return {key: {'rounds': len(values), 'samples_us': values, 'median_us': statistics.median(values), 'min_us': min(values),
                  'max_us': max(values), 'mean_us': statistics.mean(values),
                  'sample_stdev_us': statistics.stdev(values) if len(values) > 1 else None}
            for key, values in samples.items()}


def correctness_preflight(torch, triton, current, reference, graph_for, selected, seed, result):
    # Untimed, phase-independent contract checks. These are not policy data.
    result['correctness_preflight'] = records = []
    splits = sorted({1} | {s for _, case in selected for s in case['candidate_splits']})
    for dtype in sorted({case['dtype'] for _, case in selected}):
        for allocated in (64, 512):
            case = {'batch': 2, 'sq': 4, 'hq': 8, 'hk': 2, 'dtype': dtype,
                    'allocated_length': allocated, 'live_lengths': [0, 1], 'causal': True}
            inputs = make_inputs(torch, case, seed)
            expected = reference.reference(*inputs, causal=True)
            for split in splits:
                candidate = make_pipeline(torch, triton, current, inputs, case, split)
                candidate['full']()
                eager = check_result(torch, (candidate['out'], candidate['lse']), expected, dtype)
                graph = graph_for(candidate['full'], 1)
                graph.replay()
                torch.cuda.synchronize()
                replay = check_result(torch, (candidate['out'], candidate['lse']), expected, dtype)
                records.append({'spec': case, 'seed': seed, 'splits': split, 'route': candidate['route'],
                                'eager': eager, 'graph_replay': replay, 'timed': False})
                del graph, candidate


def run(args, protocol, protocol_raw, frozen):
    try:
        import torch
        import triton
    except ImportError as exc:
        raise RuntimeError(f'Existing Torch/Triton environment unavailable: {exc}') from exc
    selector = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not selector or ',' in selector or selector == '-1':
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to exactly one available GPU')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable; no benchmark was run')
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError('One selected SM120 GPU is required')
    initial_state = gpu_state(selector)
    require_unshared(initial_state)
    from flash_mla_old import sm120 as current
    from flash_mla_old import flash_mla_with_kvcache as public_forward, get_mla_metadata as public_metadata
    if Path(current.__file__).resolve() != (REPO / 'flash_mla_old/sm120.py').resolve():
        raise RuntimeError('Imported runtime is outside this checkout; fix the existing editable link')
    helper_path, reference_path = REPO / 'benchmarks/benchmark_sm120_reduce.py', REPO / 'tests/test_sm120.py'
    helper = import_file('_sm120_policy_timing_helper', helper_path)
    reference = import_file('_sm120_policy_reference', reference_path)
    paths = {'runtime': Path(current.__file__).resolve(), 'reference': reference_path, 'timing_helper': helper_path,
             'harness': Path(__file__).resolve(), 'public_interface': REPO / 'flash_mla_old/flash_mla_interface.py'}
    hashes = {name: sha256(path.read_bytes()) for name, path in paths.items()}
    output = Path(args.output).expanduser().resolve()
    if (Path.home() / 'nhl_project').resolve() not in output.parents:
        raise ValueError('--output must be a fresh directory below ~/nhl_project')
    if output.exists():
        raise ValueError('--output already exists; choose a fresh directory')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'protocol.json').write_bytes(protocol_raw)
    if frozen:
        (output / 'freeze_record.json').write_bytes(frozen['_record_bytes'])
        (output / 'frozen_policy_artifact').write_bytes(frozen['_policy_bytes'])
    for name, path in paths.items():
        (output / f'source_{name}.py').write_bytes(path.read_bytes())
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    selected = [(i, case) for i, case in enumerate(protocol['cases']) if args.phase == 'all' or case['phase'] == args.phase]
    result = {'schema_version': 1, 'experiment': 'sm120-fixed-split-policy', 'status': 'running',
              'started_utc': utc_now(), 'protocol_id': protocol['protocol_id'], 'protocol_sha256': sha256(protocol_raw),
              'phase': args.phase, 'seed': protocol['seed'], 'selected_case_ids': [case['id'] for _, case in selected],
              'case_order': 'protocol order filtered by phase', 'rounds': args.rounds, 'repetitions': args.repetitions,
              'cpu_threads': args.cpu_threads, 'source_paths': {name: str(path) for name, path in paths.items()},
              'source_hashes': hashes, 'git_head': run_command(['git', 'rev-parse', 'HEAD'], cwd=REPO),
              'git_status': run_command(['git', 'status', '--short'], cwd=REPO),
              'git_diff': run_command(['git', 'diff', '--binary', 'HEAD', '--', 'flash_mla_old', 'tests/test_sm120.py', 'benchmarks/benchmark_sm120_reduce.py'], cwd=REPO),
              'environment': {'python': sys.version, 'python_executable': sys.executable, 'torch': torch.__version__,
                              'triton': triton.__version__, 'cuda': torch.version.cuda, 'gpu': props.name,
                              'capability': [props.major, props.minor], 'sm_count': props.multi_processor_count,
                              'total_memory_bytes': props.total_memory, 'CUDA_VISIBLE_DEVICES': selector,
                              'process_id': os.getpid(), 'tf32_matmul': False, 'tf32_cudnn': False},
              'freeze_record': {k: v for k, v in frozen.items() if not k.startswith('_')} if frozen else None,
              'tolerances': TOLERANCES, 'gpu_before': initial_state, 'cases': [],
              'timing_boundary': 'CUDA events per call in repeated graphs; preallocated producer/finish/full separately; excludes allocations, metadata, JIT, H2D, reference; warm resident inputs',
              'ordering_rule': 'separate full-primary, producer, finish groups; protocol candidate order; requested rounds rounded up to a multiple of group size; rotate each round and reverse alternate complete cycles',
              'limitations': ['GPU guards are observations, not reservations; jobs between guards or graphics workloads may interfere.',
                              'Repeated graphs reuse warm inputs and scratch; these are not cold HBM or model/serving timings.',
                              'Finish-only reads fixed resident partials; separate stage medians need not sum to full median.',
                              'Freeze hashes establish artifact identity, not proof that holdout outcomes were never viewed.']}
    save_result(output, result)
    try:
        with torch.inference_mode():
            correctness_preflight(torch, triton, current, reference, helper.graph_for, selected, protocol['seed'], result)
            save_result(output, result)
            for original_index, case in selected:
                before = gpu_state(selector)
                require_unshared(before)
                seed = case.get('seed', protocol['seed'] + original_index)
                record = {'case_id': case['id'], 'phase': case['phase'], 'spec': case, 'input_seed': seed,
                          'status': 'running', 'gpu_before': before, 'candidates': []}
                result['cases'].append(record)
                save_result(output, result)
                inputs = make_inputs(torch, case, seed)
                q, cache, table, lengths = inputs
                expected = reference.reference(*inputs, scale=case.get('softmax_scale', 576 ** -0.5), causal=case['causal'])
                public_result, observed = observe_public(torch, current, public_forward, public_metadata, inputs, case)
                production_splits = observed['splits_argument']
                row_tiles = triton.cdiv(case['sq'] * (case['hq'] // case['hk']), 16)
                formula_splits = min(32, triton.cdiv(case['allocated_length'], 256),
                    max(1, triton.cdiv(2 * props.multi_processor_count, case['batch'] * case['hk'] * row_tiles)))
                if production_splits != formula_splits or production_splits not in case['candidate_splits']:
                    raise AssertionError('Candidates must include observed production splits, matching the current scheduler formula')
                record.update({'production_splits': production_splits, 'production_launch': observed,
                    'public_reference_check': check_result(torch, public_result, expected, case['dtype']),
                    'candidate_order': case['candidate_splits'],
                    'geometry': {'rows': case['batch'] * case['sq'] * case['hq'], 'row_tiles': row_tiles,
                        'allocated_length': case['allocated_length'], 'max_pages': table.shape[1],
                        'actual_live_lengths': lengths.cpu().tolist(), 'physical_cache_pages': cache.shape[0],
                        'q_shape': list(q.shape), 'q_stride': list(q.stride()), 'cache_shape': list(cache.shape),
                        'cache_stride': list(cache.stride()), 'table_shape': list(table.shape), 'table_stride': list(table.stride()),
                        'table_sha256': sha256(table.cpu().contiguous().numpy().tobytes()),
                        'page_layout': 'seeded permutation, disjoint pages, inactive entries -1, last-page tail NaN'}})
                pipelines = []
                for splits in case['candidate_splits']:
                    candidate = make_pipeline(torch, triton, current, inputs, case, splits)
                    candidate['full']()
                    checks = {'eager': check_result(torch, (candidate['out'], candidate['lse']), expected, case['dtype'])}
                    if splits == production_splits:
                        same = bool(torch.equal(candidate['out'], public_result[0]) and torch.equal(candidate['lse'], public_result[1]))
                        if not same:
                            raise AssertionError('Production-selected pipeline must exactly match public output/LSE')
                        if observed['direct'] != (candidate['route'] == 'direct') or observed['finish_launch_calls'] != int(candidate['route'] != 'direct'):
                            raise AssertionError('Research route differs from observed production route')
                        checks['public_api_bitwise_equal'] = True
                    record['candidates'].append({'splits': splits, 'route': candidate['route'],
                                                 'launch': candidate['launch'], 'correctness': checks, 'timings': {}})
                    pipelines.append(candidate)
                    save_result(output, result)
                timing = paired_graph_times(torch, helper.graph_for, pipelines, args.rounds, args.repetitions, record)
                for candidate, saved in zip(pipelines, record['candidates']):
                    saved['correctness']['graph_replay'] = check_result(torch, (candidate['out'], candidate['lse']), expected, case['dtype'])
                    if candidate['splits'] == production_splits:
                        same = bool(torch.equal(candidate['out'], public_result[0]) and torch.equal(candidate['lse'], public_result[1]))
                        if not same:
                            raise AssertionError('Captured production-selected output/LSE differ from public API')
                        saved['correctness']['graph_public_api_bitwise_equal'] = True
                    for component in ('producer', 'finish', 'full'):
                        saved['timings'][component] = timing.get((candidate['splits'], component))
                    saved['finish_kernel_absent'] = candidate['route'] == 'direct'
                record['gpu_after'] = gpu_state(selector)
                require_unshared(record['gpu_after'])
                record['status'] = 'complete'
                record['cuda_memory_allocated_bytes'] = torch.cuda.memory_allocated()
                record['cuda_memory_reserved_bytes'] = torch.cuda.memory_reserved()
                save_result(output, result)
                print(f"Completed {case['id']} ({case['phase']}), correctness passed", flush=True)
                del pipelines, candidate, inputs, q, cache, table, lengths, expected, public_result
        result['gpu_after'] = gpu_state(selector)
        require_unshared(result['gpu_after'])
        result['source_hashes_after'] = {name: sha256(path.read_bytes()) for name, path in paths.items()}
        if hashes != result['source_hashes_after']:
            raise RuntimeError('Sources changed during this run; this is not a frozen-source experiment')
        result['status'], result['completed_utc'] = 'complete', utc_now()
        save_result(output, result)
    except BaseException as exc:
        result.update(status='failed', failed_utc=utc_now(), error={'type': type(exc).__name__, 'message': str(exc)})
        save_result(output, result)
        raise
    print(f'Raw results: {output / "results.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', help='Fresh run directory below ~/nhl_project')
    parser.add_argument('--phase', choices=('train', 'holdout', 'all'), default='train')
    parser.add_argument('--freeze-record')
    parser.add_argument('--freeze-record-sha256')
    parser.add_argument('--rounds', type=positive_int, default=15)
    parser.add_argument('--repetitions', type=positive_int, default=128)
    parser.add_argument('--cpu-threads', type=positive_int, default=2)
    parser.add_argument('--validate-protocol-only', action='store_true')
    args = parser.parse_args()
    try:
        protocol, raw = read_protocol(Path(args.protocol).expanduser().resolve(strict=True))
        frozen = validate_freeze(args, sha256(raw))
        selected = [case for case in protocol['cases'] if args.phase == 'all' or case['phase'] == args.phase]
        if not selected:
            raise ValueError('Selected phase has no cases')
        if args.validate_protocol_only:
            print(json.dumps({'protocol_id': protocol['protocol_id'], 'protocol_sha256': sha256(raw),
                              'phase': args.phase, 'selected_case_ids': [case['id'] for case in selected],
                              'freeze_validated': frozen is not None}, indent=2))
            return 0
        if not args.output:
            parser.error('--output is required unless --validate-protocol-only is selected')
        run(args, protocol, raw, frozen)
        return 0
    except (ValueError, OSError, RuntimeError, AssertionError, ImportError) as exc:
        print(f'Experiment stopped: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
