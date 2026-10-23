"""Summarize the fixed research suites using only Python's standard library.

Prints Markdown to stdout, so redirect into the project-local run directory.
An affine bytes-only latency model is a diagnostic, not a TMA performance model.
"""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import statistics


def affine_bytes_diagnostic(cases, backend):
    # Hold out complete split counts, including the 8/9 dispatch boundary.
    train = [c for c in cases if c['splits'] in (2, 4, 16, 32)]
    held = [c for c in cases if c['splits'] in (8, 9, 24)]
    xs = [c['logical_read_write_bytes'] / 2**20 for c in train]
    ys = [c['timings'][backend]['median_us'] for c in train]
    xm, ym = statistics.mean(xs), statistics.mean(ys)
    slope = sum((x-xm)*(y-ym) for x,y in zip(xs,ys)) / sum((x-xm)**2 for x in xs)
    intercept = ym - slope*xm
    errors = [abs(intercept + slope*c['logical_read_write_bytes']/2**20 - c['timings'][backend]['median_us']) / c['timings'][backend]['median_us'] for c in held]
    return intercept, slope, statistics.mean(errors), max(errors)


def render(low, grid):
    expected_low = {(b,1,128,1,l,'bf16') for b in (1,2,4,8) for l in (64,256,1024,4096,16384,32768)}
    expected_grid = {(r,s) for r in (128,256,512,1024) for s in (1,2,4,8,9,16,24,32)}
    assert len(low['cases']) == 24 and {tuple(c['shape']) for c in low['cases']} == expected_low
    assert len(grid['cases']) == 32 and {(c['rows'],c['splits']) for c in grid['cases']} == expected_grid
    assert low['completed_utc'] and grid['completed_utc']
    assert low['reference_checked'] and all(c['reference_checks'] for c in low['cases']+grid['cases'])
    assert low['candidate_sha256'] == grid['candidate_sha256']
    gm = statistics.geometric_mean
    retained = [c for c in low['cases'] if 'reduce_speedup' in c]
    lines = [
        '# RTX low-batch characterization: ' + (datetime.fromisoformat(low['started_utc'].replace('Z','+00:00')) + timedelta(hours=8)).strftime('%d %B %Y'), '',
        'This compares the candidate RTX backend with the frozen initial RTX port. These are operator '
        'measurements on RTX 6000D, not Hopper/TMA or complete-model results.', '',
        '## Results', '',
        f"- Full-forward geometric-mean speedup: **{gm(c['full_speedup'] for c in low['cases']):.3f}x** across 24 cases.",
        f"- Retained reducer geometric-mean speedup: **{gm(c['reduce_speedup'] for c in retained):.3f}x** across {len(retained)} cases; eight direct-output cases eliminate this launch and are excluded.",
        f"- Controlled multi-split reducer geometric-mean speedup: **{gm(c['speedup'] for c in grid['cases'] if c['splits'] > 1):.3f}x** across 28 cases.",
        f"- {sum(c['full_speedup'] > 1 for c in low['cases'])} of 24 full-forward medians improved, spanning {min(c['full_speedup'] for c in low['cases']):.3f}x to {max(c['full_speedup'] for c in low['cases']):.3f}x. Small improvements are single-session observations, not cross-session confidence claims.",
        '- Both implementations passed the independent FP32 attention reference on all 24 shapes, and FP64 weighted-reduction checks on all 32 synthetic cases.',
        f"- Candidate maximum output error against FP32: {max(c['reference_checks']['after']['max_abs_output_error'] for c in low['cases']):.8f}; maximum natural-log LSE error: {max(c['reference_checks']['after']['max_abs_lse_error'] for c in low['cases']):.3g}.", '',
        '## Full-forward sweep', '',
        'Fixed Sq=1, Hq=128, Hkv=1, QK=576, V=512, 64-token pages, BF16, causal. '
        'Values are medians in microseconds. Reducer columns come from a separate paired '
        'measurement; they must not be added to full-forward timings or treated as a profiled time fraction.', '',
        '| Batch | KV length | Splits | Full before | Full after | Speedup | Reduce before | Reduce after |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for c in low['cases']:
        t=c['timings']; m=c.get('merge_comparison')
        rb=m['before']['median_us'] if m else t['reduce_before']['median_us']
        ra=f"{m['after']['median_us']:.3f}" if m else 'eliminated'
        lines.append(f"| {c['shape'][0]} | {c['shape'][4]} | {c['splits']} | {t['full_before']['median_us']:.3f} | {t['full_after']['median_us']:.3f} | {c['full_speedup']:.3f}x | {rb:.3f} | {ra} |")
    lines += ['', 'Long-KV forward time is dominated by attention in this implementation. '
              'For example, B=1, KV=32768 improves only about one microsecond despite a faster reducer. '
              'The short-KV direct-output path and small split counts offer more operator-level benefit. '
              'This does not establish a universal combine fraction for FlashMLA.', '',
              '## Controlled split sweep', '',
              'The split count is forced only for standalone reducer inputs, independent of sequence '
              'length and the attention scheduler. FP32 partials and base-2 LSE feed BF16 output. '
              'All 32 cases include an empty row; multi-split cases also include masked partials. '
              'The S=1 case times the copy path even though short-context forward bypasses it.', '',
              '| Rows | S=1 | S=2 | S=4 | S=8 | S=9 | S=16 | S=24 | S=32 |',
              '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for rows in (128,256,512,1024):
        cells=[f"{next(c['speedup'] for c in grid['cases'] if c['rows']==rows and c['splits']==s):.3f}x" for s in (1,2,4,8,9,16,24,32)]
        lines.append('| '+str(rows)+' | '+' | '.join(cells)+' |')
    regressions=[c for c in grid['cases'] if c['speedup']<1]
    lines += ['', 'Measured regressions are retained:' if regressions else 'No median regression was measured in this grid.']
    for c in regressions:
        before=c['timings']['before']['median_us']; after=c['timings']['after']['median_us']
        lines.append(f"- Rows={c['rows']}, S={c['splits']}: {before:.3f} -> {after:.3f} us ({(after/before-1)*100:.2f}% higher latency, {after-before:.3f} us).")
    lines += ['', 'Standalone S=1 copy measurements describe a different path from short-context '
              'direct output. Keep both paths visible when choosing future policies.', '',
              '## Initial bytes-only model and held-out checks', '',
              'As a diagnostic, fit `latency_us = a + b * logical_MiB` independently for each backend '
              'on S={2,4,16,32}; hold out S={8,9,24} at every row count. S=1 is excluded because it has '
              'a different algorithm. Logical bytes count one read of each partial/LSE and one output/LSE '
              'write: `4*R*S*(512+1) + R*(2*512+4)`. They are not measured HBM bytes. '
              'This exploratory split was selected during analysis and uses 16 training and 12 held-out configurations per backend.', '',
              '| Backend | a (us) | b (us/MiB) | Held-out mean absolute percentage error | Maximum error |',
              '|---|---:|---:|---:|---:|']
    for backend in ('before','after'):
        a,b,mape,worst=affine_bytes_diagnostic(grid['cases'],backend)
        lines.append(f'| {backend} | {a:.3f} | {b:.3f} | {mape*100:.1f}% | {worst*100:.1f}% |')
    lines += ['', 'These errors measure predictive fit on this one experiment, not uncertainty across '
              'GPU sessions. The grid includes launch-wave effects, rounded split extents and a dispatch '
              'change at S=8/9. A useful next model should account for these terms and validate on new '
              'sessions and row counts. This affine diagnostic does not predict Hopper TMA break-even.', '',
              '## Reproduction and provenance', '',
              f"- Initial RTX baseline commit: `{low['baseline_commit']}`.",
              f"- Checkout HEAD during measurement: `{low['candidate_commit']}`; benchmark edits were uncommitted and identified by hashes in the raw records.",
              f"- Runtime source SHA-256: `{low['candidate_sha256']}`; identical in the two input experiment records.",
              f"- GPU: {low['gpu']}, {low['sm_count']} SMs; Torch {low['torch']}, Triton {low['triton']}, CUDA runtime {low['cuda']}.",
              f"- UTC intervals: {low['started_utc']} to {low['completed_utc']} (forward); {grid['started_utc']} to {grid['completed_utc']} (split grid). The title uses Asia/Shanghai time.",
              f"- Each variant uses 128 graph-batched calls and {low['rounds']} timing rounds for forward, {grid['rounds']} for the split grid, with rotating/reversed order, warm resident buffers and CUDA events. The benchmark does not alter clocks/power settings.",
              '- Timing excludes JIT, metadata preparation and host-to-device transfer. The synthetic reducer grid is cache-resident repeated work; effective logical bandwidth is not device-memory bandwidth.',
              '- Shared-server GPU visibility is not a reservation. Check for competing jobs before each run; the saved results represent one session per suite.', '',
              'Activate and check an available GPU before running these commands from the repository. '
              'Both benchmark programs require a new output directory under `~/nhl_project` and refuse '
              'to reuse an existing directory.', '', '```bash',
              'source scripts/rtx6000d_env.sh',
              'nvidia-smi',
              '# Set this to a GPU verified idle immediately before the run.',
              'export CUDA_VISIBLE_DEVICES=0',
              'RUN_ROOT="$HOME/nhl_project/moore_tma_pj/docs/runs/low-batch-$(date +%Y%m%d-%H%M%S)"',
              'python benchmarks/benchmark_sm120_reduce.py --suite low-batch --check-reference \\',
              '  --rounds 15 --output "$RUN_ROOT/low_batch"',
              'python benchmarks/benchmark_sm120_split_grid.py --rounds 15 --output "$RUN_ROOT/split_grid"',
              'python benchmarks/summarize_sm120_research.py \\',
              '  --low-batch "$RUN_ROOT/low_batch/results.json" \\',
              '  --split-grid "$RUN_ROOT/split_grid/results.json" > "$RUN_ROOT/summary.md"',
              '```', '',
              'The input JSON records contain all raw timing samples and source hashes. '
              'The repository documentation index links the published result snapshots and the '
              'separate Hopper research-readiness audit.', '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--low-batch',type=Path,required=True)
    parser.add_argument('--split-grid',type=Path,required=True)
    args=parser.parse_args()
    print(render(json.loads(args.low_batch.read_text()),json.loads(args.split_grid.read_text())))


if __name__=='__main__':
    main()
