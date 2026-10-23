"""Run directly: python tests/test_sm120.py (no pytest dependency)."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import unittest
from unittest import mock

import torch

from flash_mla_old import flash_mla_with_kvcache, get_mla_metadata
from flash_mla_old import flash_mla_interface as interface


@torch.inference_mode()
def reference(q, cache, table, lengths, scale=None, causal=False):
    """Independent FP32 paged attention with bottom-right causal masking."""
    b, sq, hq, d = q.shape
    hk = cache.shape[2]
    scale = d ** -0.5 if scale is None else scale
    out = torch.zeros((b, sq, hq, 512), device=q.device, dtype=torch.float32)
    lse = torch.full((b, hq, sq), -float('inf'), device=q.device, dtype=torch.float32)
    for i, length in enumerate(lengths.tolist()):
        if length == 0:
            continue
        k = cache[table[i, :math.ceil(length / 64)].long()].reshape(-1, hk, d)[:length].float()
        query = q[i].float().reshape(sq, hk, hq // hk, d)
        scores = torch.einsum('shgd,khd->hgsk', query, k) * scale
        if causal:
            visible = torch.arange(length, device=q.device)[None, :] <= length - sq + torch.arange(sq, device=q.device)[:, None]
            scores.masked_fill_(~visible[None, None], -float('inf'))
        lse[i] = scores.logsumexp(-1).reshape(hq, sq)
        probs = torch.nan_to_num(scores.softmax(-1), nan=0.0)
        out[i] = torch.einsum('hgsk,khd->shgd', probs, k[..., :512]).reshape(sq, hq, 512)
    return out, lse


def make_inputs(lengths, sq, hq, hk, dtype, strided=False):
    b, max_length = len(lengths), max(1, max(lengths))
    max_pages = math.ceil(max_length / 64)
    device = torch.device('cuda:0')
    q = torch.randn(b, sq, hq, 584 if strided else 576, dtype=dtype, device=device)[..., :576]
    cache = torch.randn(b * max_pages, 64, hk, 584 if strided else 576, dtype=dtype, device=device)[..., :576]
    table = torch.randperm(b * max_pages, device=device, dtype=torch.int64).to(torch.int32).reshape(b, max_pages)
    if strided:
        wide = torch.empty(b, max_pages + 3, dtype=torch.int32, device=device)
        wide[:, :max_pages] = table
        table = wide[:, :max_pages]
    # Poison unused tokens and page entries to catch padding/masking mistakes.
    for i, length in enumerate(lengths):
        for token in range(length, max_pages * 64):
            cache[table[i, token // 64], token % 64] = float('nan')
        table[i, math.ceil(length / 64):] = -1
    lengths_t = torch.tensor(lengths, dtype=torch.int32, device=device)
    metadata = get_mla_metadata(lengths_t, sq * hq // hk, hk)
    return q, cache, table, lengths_t, metadata


def forward(inputs, causal=False, scale=None):
    q, cache, table, lengths, metadata = inputs
    return flash_mla_with_kvcache(q, cache, table, lengths, 512, *metadata,
                                softmax_scale=scale, causal=causal)


METRICS = []


def check(inputs, causal=False, scale=None, result=None):
    q, cache, table, lengths, _ = inputs
    out, lse = forward(inputs, causal, scale) if result is None else result
    expected, expected_lse = reference(q, cache, table, lengths, scale, causal)
    assert out.dtype == q.dtype and out.shape == expected.shape
    assert lse.dtype == torch.float32 and lse.shape == expected_lse.shape
    torch.testing.assert_close(out.float(), expected, atol=0.006 if q.dtype == torch.bfloat16 else 0.001, rtol=0.02 if q.dtype == torch.bfloat16 else 0.005)
    torch.testing.assert_close(lse, expected_lse, atol=2e-4, rtol=2e-5)
    cosine_error = (out.float() - expected).square().sum() / (out.float().square().sum() + expected.square().sum()).clamp_min(1e-12)
    assert cosine_error.item() < 1e-5
    METRICS.append({'shape': list(q.shape), 'lengths': lengths.tolist(), 'dtype': str(q.dtype),
                    'causal': causal, 'scale': scale, 'max_abs_output_error': (out.float()-expected).abs().max().item(),
                    'cosine_error': cosine_error.item()})


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (12, 0), 'SM120 GPU required')
class SM120Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(20261023)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_num_threads(2)

    @torch.inference_mode()
    def test_numerics(self):
        cases = [([1], 1, 1, 1, False, None, False),
                 ([63, 64, 65], 1, 16, 1, True, None, False),
                 ([511, 1025], 2, 32, 2, True, None, False),
                 ([257, 129], 3, 12, 3, False, 0.075, True),
                 ([0, 1, 2, 65], 4, 8, 2, True, None, False),
                 ([0, 17], 1, 16, 1, False, None, False),
                 ([4097], 2, 128, 1, True, None, False),
                 ([32768], 1, 128, 1, True, None, False),
                 ([257], 1, 16, 1, False, 0.0, False),
                 ([257], 2, 16, 1, True, -0.05, False)]
        for dtype in (torch.bfloat16, torch.float16):
            for lengths, sq, hq, hk, causal, scale, strided in cases:
                with self.subTest(dtype=dtype, lengths=lengths, sq=sq, hq=hq, hk=hk, causal=causal):
                    check(make_inputs(lengths, sq, hq, hk, dtype, strided), causal, scale)

    @torch.inference_mode()
    def test_graph_and_live_lengths(self):
        inputs = make_inputs([513, 129], 2, 16, 1, torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward(inputs, True)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = forward(inputs, True)
        graph.replay()
        check(inputs, True, result=captured)
        # Reuse metadata and the graph with shorter live lengths; then grow back.
        inputs[3].copy_(torch.tensor([65, 0], device='cuda', dtype=torch.int32))
        graph.replay()
        check(inputs, True, result=captured)
        inputs[3].copy_(torch.tensor([513, 129], device='cuda', dtype=torch.int32))
        graph.replay()
        check(inputs, True, result=captured)

    @torch.inference_mode()
    def test_validation(self):
        q, cache, table, lengths, metadata = make_inputs([65], 1, 16, 1, torch.bfloat16)
        with self.assertRaisesRegex(ValueError, 'dimension'):
            flash_mla_with_kvcache(q[..., :512], cache, table, lengths, 512, *metadata)
        with self.assertRaisesRegex(ValueError, 'FP16 or BF16'):
            flash_mla_with_kvcache(q.float(), cache, table, lengths, 512, *metadata)
        with self.assertRaisesRegex(ValueError, 'same CUDA device'):
            flash_mla_with_kvcache(q, cache, table.cpu(), lengths, 512, *metadata)
        with self.assertRaisesRegex(ValueError, 'metadata'):
            flash_mla_with_kvcache(q, cache, table, lengths, 512, torch.empty((1, 8), device=q.device, dtype=torch.int32), metadata[1])
        with self.assertRaisesRegex(ValueError, 'int32'):
            get_mla_metadata(lengths.long(), 16, 1)
        with self.assertRaisesRegex(ValueError, 'finite'):
            flash_mla_with_kvcache(q, cache, table, lengths, 512, *metadata, softmax_scale=float('nan'))


class DispatchTests(unittest.TestCase):
    def test_cpu_and_unsupported_device(self):
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            get_mla_metadata(torch.tensor([1], dtype=torch.int32), 16, 1)
        tensor = mock.Mock(is_cuda=True, device=torch.device('cuda:0'))
        with mock.patch.object(torch.cuda, 'get_device_capability', return_value=(8, 9)):
            with self.assertRaisesRegex(RuntimeError, 'Unsupported'):
                interface._backend_for(tensor)

    def test_hopper_dispatch_keeps_native_arguments(self):
        tensor = mock.Mock(is_cuda=True, device=torch.device('cuda:0'), shape=(1, 1, 16, 576))
        ext = mock.Mock()
        ext.get_mla_metadata.return_value = ('meta', 'splits')
        ext.fwd_kvcache_mla.return_value = ('out', 'lse')
        with mock.patch.object(torch.cuda, 'get_device_capability', return_value=(9, 0)), mock.patch.object(torch.cuda, 'device'), mock.patch.object(interface, '_hopper_extension', return_value=ext):
            self.assertEqual(get_mla_metadata(tensor, 16, 1), ('meta', 'splits'))
            self.assertEqual(flash_mla_with_kvcache(tensor, 'cache', 'table', 'lengths', 512, 'meta', 'splits', causal=True), ('out', 'lse'))
        ext.fwd_kvcache_mla.assert_called_once_with(tensor, 'cache', None, 512, 'lengths', 'table', 576**-0.5, True, 'meta', 'splits')


def benchmark():
    inputs = make_inputs([32768], 1, 128, 1, torch.bfloat16)
    with torch.inference_mode():
        check(inputs, True)
        for _ in range(10):
            forward(inputs, True)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            forward(inputs, True)
        end.record()
        end.synchronize()
        eager_ms = start.elapsed_time(end) / 100
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward(inputs, True)
        for _ in range(10):
            graph.replay()
        start.record()
        for _ in range(100):
            graph.replay()
        end.record()
        end.synchronize()
        return {'case': 'B1 Sq1 Hq128 Hkv1 KV32768 BF16 causal',
                'eager_mean_ms': eager_ms, 'graph_mean_ms': start.elapsed_time(end) / 100,
                'iterations': 100,
                'boundary': 'both kernels; eager includes Python/allocation gaps, graph uses captured allocations; excludes metadata, JIT, reference'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report')
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    started = time.time()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'success': result.wasSuccessful(), 'tests': result.testsRun,
              'skipped': len(result.skipped), 'seconds': time.time() - started,
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              'metrics': METRICS}
    if args.benchmark and result.wasSuccessful() and not result.skipped:
        report['benchmark'] = benchmark()
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'metrics'}, indent=2))
    sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
