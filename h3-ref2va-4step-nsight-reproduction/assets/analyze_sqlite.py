#!/usr/bin/env python3
"""Read-only Nsight Systems analyzer; no CUDA imports or GPU work.

All report times are seconds relative to measured_run NVTX start. Raw metric
CSV retains original timestamp fields too. Metric samples are treated as trailing
interval averages (previous timestamp, timestamp]; first samples have no inferred
coverage. Every subsequent interval is retained, including sampling gaps.
"""
import argparse
from array import array
import gzip
import bisect
import collections
import csv
import heapq
import json
import math
import re
import sqlite3
import statistics
from pathlib import Path

NS = 1e9


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def union(intervals):
    result = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def duration(intervals):
    return sum(b - a for a, b in union(intervals)) / NS


def family(name):
    n = name.lower()
    for label, pattern in [
        ('attention', r'flash|fmha|attention|attn|sage|softmax'),
        ('matrix_multiply', r'gemm|gemv|cutlass|cublas|mma|matmul'),
        ('quantization', r'quant|dequant|fp8|int8|nvfp4'),
        ('convolution', r'cudnn|conv|winograd'),
        ('normalization', r'layer_norm|layernorm|rmsnorm|rms_norm|group_norm'),
        ('reduction', r'reduce|reduction|scan|sum_kernel'),
        ('copy_layout', r'copy|transpose|permute|gather|scatter|cat|index'),
        ('elementwise', r'elementwise|vectorized|pointwise|activation|silu|gelu')]:
        if re.search(pattern, n):
            return label
    return 'other'


def summarize(events):
    spans = [(e['start'], e['end']) for e in events]
    times = sorted((b - a) / NS for a, b in spans)
    return {'count': len(events), 'duration_sum_s': sum(times),
            'presence_union_s': duration(spans),
            'mean_s': statistics.mean(times) if times else None,
            'p95_s': times[math.ceil(.95 * len(times)) - 1] if times else None}


def analyze(db, output, csv_path):
    con = sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True, timeout=1)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA query_only=ON')
    con.execute('BEGIN')
    schema = {r[0]: [c[1] for c in con.execute('PRAGMA table_info(' + quote(r[0]) + ')')]
              for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def rows(table, suffix=''):
        if table not in schema:
            return []
        return (dict(r) for r in con.execute('SELECT * FROM ' + quote(table) + ' ' + suffix))

    strings = {r['id']: r['value'] for r in rows('StringIds')}

    def name(row, *keys):
        for key in keys:
            v = row.get(key)
            if v is not None:
                return str(strings.get(v, v)) if isinstance(v, int) else str(v)
        return ''

    warnings = []
    nvtx = []
    for r in rows('NVTX_EVENTS'):
        if r.get('end') is None or r['end'] <= r['start']:
            continue
        r['label'] = name(r, 'text', 'textId')
        if not r['label'] and r.get('textId') is not None:
            r['label'] = name(r, 'textId')
        nvtx.append(r)
    measured = [r for r in nvtx if r['label'] == 'measured_run']
    if len(measured) != 1:
        raise RuntimeError(f'Expected one completed measured_run NVTX range; found {len(measured)}')
    root = measured[0]
    origin, stop = root['start'], root['end']
    pid = root['globalTid'] >> 24
    nvtx = [r for r in nvtx if origin <= r['start'] < stop and
            r['end'] <= stop and r.get('globalTid', 0) >> 24 == pid]
    phases = [r for r in nvtx if not r['label'].startswith('aten::') and
              (r['label'] == 'measured_run' or r['label'].startswith(('node/', 'DiT_step_')) or
               r['label'] in ('text_tokenization', 'text_encoder', 'denoising_loop', 'final_export'))]
    phases.sort(key=lambda r: (r['start'], -r['end']))
    for i, p in enumerate(phases):
        p['phase_id'] = i
    by_thread = collections.defaultdict(list)
    for r in nvtx:
        if r['label'].startswith('aten::'):
            by_thread[r['globalTid']].append(r)
    runtimes = list(rows('CUPTI_ACTIVITY_KIND_RUNTIME', 'ORDER BY start'))
    if not runtimes:
        raise RuntimeError('No CUDA runtime events; cannot project NVTX')
    # Sweep each CPU thread: the shortest containing ATen range is innermost.
    heaps, positions = {}, collections.defaultdict(int)
    for tid, rr in by_thread.items():
        rr.sort(key=lambda r: r['start'])
        heaps[tid] = []
    correlated = collections.defaultdict(list)
    for r in runtimes:
        tid = r.get('globalTid', 0)
        if tid >> 24 != pid or not origin <= r['start'] < stop:
            continue
        rr, heap = by_thread.get(tid, []), heaps.setdefault(tid, [])
        pos = positions[tid]
        while pos < len(rr) and rr[pos]['start'] <= r['start']:
            x = rr[pos]
            heapq.heappush(heap, (x['end'] - x['start'], -x['start'], pos, x['end']))
            pos += 1
        positions[tid] = pos
        expired = []
        while heap and heap[0][3] < r['end']:
            x = heapq.heappop(heap)
            if x[3] > r['start']:
                expired.append(x)
        op = rr[heap[0][2]]['label'] if heap else None
        for x in expired:
            heapq.heappush(heap, x)
        r['aten'] = re.split(r',\s*(?:seq|sizes|input_sizes|op_id)\s*=', op, maxsplit=1)[0] if op else None
        r['phases'] = [p['phase_id'] for p in phases if p['globalTid'] == tid and
                       p['start'] <= r['start'] and r['end'] <= p['end']]
        correlated[(tid >> 24, r.get('correlationId'))].append(r)

    def attach(r):
        process = r.get('globalPid')
        process = process >> 24 if process is not None else pid
        candidates = correlated.get((process, r.get('correlationId')), [])
        candidates = [x for x in candidates if x['start'] <= r['start']]
        rt = max(candidates, key=lambda x: x['start']) if candidates else None
        r['aten'] = rt['aten'] if rt else None
        r['phases'] = rt['phases'] if rt else []
        r['runtime_start_s'] = (rt['start'] - origin) / NS if rt else None
        return rt

    kernels, transfers, memsets = [], [], []
    kernel_tables = [t for t in schema if t in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL')]
    # Some versions provide aliases; choose the preferred non-empty table.
    for table in kernel_tables:
        if kernels:
            break
        for r in rows(table):
            if r['end'] <= origin or r['start'] >= stop:
                continue
            if r.get('globalPid') is not None and r['globalPid'] >> 24 != pid:
                continue
            r['name'] = name(r, 'demangledName', 'shortName', 'mangledName', 'name')
            r['family'] = family(r['name'])
            attach(r)
            kernels.append(r)
    if not kernels:
        raise RuntimeError('No measured GPU kernels yet')
    enum = {r['id']: name(r, 'label', 'name') for r in rows('ENUM_CUDA_MEMCPY_OPER')}
    for table, dest in [('CUPTI_ACTIVITY_KIND_MEMCPY', transfers), ('CUPTI_ACTIVITY_KIND_MEMSET', memsets)]:
        for r in rows(table):
            if r['end'] <= origin or r['start'] >= stop:
                continue
            if r.get('globalPid') is not None and r['globalPid'] >> 24 != pid:
                continue
            attach(r)
            r['kind'] = enum.get(r.get('copyKind'), str(r.get('copyKind', 'memset')))
            dest.append(r)

    def spans_json(spans):
        return [[(a - origin) / NS, (b - origin) / NS] for a, b in union(spans)]

    def grouped(events, key):
        groups = collections.defaultdict(list)
        for e in events:
            groups[e.get(key) or 'unattributed'].append(e)
        return sorted([{'name': k, **summarize(v)} for k, v in groups.items()],
                      key=lambda x: -x['duration_sum_s'])

    report = {'schema_version': 1, 'source': str(db.resolve()), 'schema': schema,
              'time_unit': 'seconds', 'origin': {'nvtx': 'measured_run', 'timestamp_ns': origin,
              'global_tid': root['globalTid'], 'process_id': pid & 0xffffff, 'global_process_key': pid},
              'origin_ns': origin, 'measured_end_ns': stop, 'capture_end_ns': stop,
              'capture_end_definition': 'Completed measured_run NVTX end; profiler capture may extend beyond this',
              'measured_duration_s': (stop - origin) / NS,
              'methodology': {'phase_projection': 'CUDA runtime containment on NVTX globalTid, joined to GPU activity by process and correlationId; inclusive nested phases are not additive',
              'aten': 'One innermost fully containing ATen NVTX range per correlated runtime/kernel; names stripped of shapes/sequence metadata',
              'kernel_time': 'Duration sums may overlap; presence union is elapsed time with >=1 kernel, not SM occupancy',
              'metrics': 'Device-wide trailing sample intervals; exact overlap weights, no extrapolation of first sample; weighted p95 is duration-weighted nearest rank; intervals spanning gaps remain included',
              'warmup': 'Capture starts after warmup; only measured_run time window included'},
              'warnings': warnings, 'kernels': {'totals': summarize(kernels),
              'families': grouped(kernels, 'family'), 'full_names': grouped(kernels, 'name'),
              'presence_union_intervals_s': spans_json([(k['start'], k['end']) for k in kernels]),
              'presence_union_by_device': {str(d): spans_json([(k['start'], k['end']) for k in kernels if k.get('deviceId') == d]) for d in {k.get('deviceId') for k in kernels}}},
              'aten_operators': grouped(kernels, 'aten'), 'phases': []}
    details = list(rows('ANALYSIS_DETAILS'))
    report['capture_metadata'] = {'analysis_details': details,
        'session_start_time': list(rows('TARGET_INFO_SESSION_START_TIME'))}
    if details:
        report['capture_end_ns'] = max(r['duration'] for r in details)
        report['capture_end_definition'] = 'ANALYSIS_DETAILS.duration in the relative SQLite timestamp domain; raw startTime/stopTime retained in capture_metadata'
    unmatched = sum(k['runtime_start_s'] is None for k in kernels)
    report['attribution'] = {'kernel_count': len(kernels), 'runtime_correlated': len(kernels) - unmatched,
                             'unmatched_runtime': unmatched, 'aten_attributed': sum(k['aten'] is not None for k in kernels)}
    if unmatched:
        warnings.append(f'{unmatched} kernels have no matching measured runtime call')
    for p in phases:
        ks = [k for k in kernels if p['phase_id'] in k['phases']]
        ts = [k for k in transfers if p['phase_id'] in k['phases']]
        gpu = ks + ts + [k for k in memsets if p['phase_id'] in k['phases']]
        bounds = [min(x['start'] for x in gpu), max(x['end'] for x in gpu)] if gpu else None
        report['phases'].append({'id': p['phase_id'], 'name': p['label'],
            'cpu_start_s': (p['start'] - origin) / NS, 'cpu_end_s': (p['end'] - origin) / NS,
            'cpu_duration_s': (p['end'] - p['start']) / NS,
            'gpu_projected_start_s': (bounds[0] - origin) / NS if bounds else None,
            'gpu_projected_end_s': (bounds[1] - origin) / NS if bounds else None,
            'gpu_projected_span_s': (bounds[1] - bounds[0]) / NS if bounds else 0,
            'gpu_activity_union_intervals_s': spans_json([(k['start'], k['end']) for k in gpu]),
            'kernels': summarize(ks), 'kernel_families': grouped(ks, 'family'),
            'aten_operators': grouped(ks, 'aten'), 'transfer_bytes': sum(t.get('bytes', 0) for t in ts)})
    def transfer_report(events):
        return {'totals': {**summarize(events), 'bytes': sum(e.get('bytes', 0) for e in events)},
                'by_kind': [{**g, 'bytes': sum(e.get('bytes', 0) for e in events if e['kind'] == g['name'])} for g in grouped(events, 'kind')],
                'timeline': [{'start_s': (e['start'] - origin) / NS, 'end_s': (e['end'] - origin) / NS,
                             **{k: e.get(k) for k in ('kind', 'bytes', 'deviceId', 'streamId', 'correlationId', 'srcKind', 'dstKind', 'srcDeviceId', 'dstDeviceId', 'phases', 'aten', 'runtime_start_s')}} for e in sorted(events, key=lambda e: e['start'])]}
    report['cuda_transfers'] = transfer_report(transfers)
    report['cuda_memsets'] = transfer_report(memsets)
    reps = []
    for fam in report['kernels']['families']:
        names = grouped([k for k in kernels if k['family'] == fam['name']], 'name')[:3]
        for n in names:
            candidates = sorted([k for k in kernels if k['name'] == n['name']], key=lambda k: k['end'] - k['start'])
            e = candidates[len(candidates) // 2]
            reps.append({'family': fam['name'], 'full_name': n['name'], **{k: v for k, v in n.items() if k != 'name'},
                         'representative': {'start_s': (e['start'] - origin) / NS, 'duration_s': (e['end'] - e['start']) / NS,
                         **{k: e.get(k) for k in ('deviceId', 'streamId', 'correlationId', 'gridX', 'gridY', 'gridZ', 'blockX', 'blockY', 'blockZ', 'phases', 'aten')}},
                         'ncu_kernel_name_base': 'demangled', 'ncu_kernel_regex': '^' + re.escape(n['name']) + '$'})
    report['ncu_candidates'] = reps
    dit3 = {p['phase_id'] for p in phases if p['label'] == 'DiT_step_3'}
    step_kernels = [k for k in kernels if dit3.intersection(k['phases'])]
    dit_names = grouped(step_kernels, 'name')
    report['ncu_dit_step_3_candidates'] = []
    for n in dit_names[:30]:
        es = sorted([k for k in step_kernels if k['name'] == n['name']], key=lambda k: k['start'])
        e = es[0]
        report['ncu_dit_step_3_candidates'].append({**n, 'full_name': n['name'], 'family': e['family'],
            'first_launch_start_s': (e['start'] - origin) / NS, 'first_launch_duration_s': (e['end'] - e['start']) / NS,
            'regex': '^' + re.escape(n['name']) + '$'})
    print(json.dumps({'ncu_dit_step_3_candidates': report['ncu_dit_step_3_candidates'][:8]}, indent=2), flush=True)
    add_memory(report, con, schema, strings, origin, stop, pid)

    print(f'Correlated {len(kernels)} kernels; processing GPU metrics', flush=True)
    add_metrics(report, con, schema, origin, stop, csv_path)
    annotate_metric_units(report)
    con.close()
    output.write_text(json.dumps(report, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'metrics_csv': str(csv_path), 'measured_duration_s': report['measured_duration_s'],
                      'kernels': report['kernels']['totals'], 'attribution': report['attribution'], 'warnings': warnings}, indent=2))


def add_memory(report, con, schema, strings, origin, stop, pid):
    tables = [t for t in schema if re.search(r'(CUPTI.*MEMORY|CUDA.*MEMORY.*USAGE)', t)]
    memory = {'tables': tables, 'events': [], 'curve': [],
              'baseline_bytes': None, 'meaning': 'Capture starts after warmup: pre-existing allocations are unknown; curve is observed allocation delta, not total device memory'}
    report['cuda_memory_allocations'] = memory
    enums = {}
    for t in schema:
        if t.startswith('ENUM_') and re.search('MEM.*OPER|MEM.*KIND', t):
            enums[t] = [dict(r) for r in con.execute('SELECT * FROM ' + quote(t))]
    memory['enums'] = enums
    for t in tables:
        for row in con.execute('SELECT * FROM ' + quote(t)):
            r = dict(row)
            timestamp = r.get('start', r.get('timestamp'))
            if timestamp is None or not origin <= timestamp <= stop:
                continue
            if r.get('globalPid') is not None and r['globalPid'] >> 24 != pid:
                continue
            e = {'source_table': t, 'time_s': (timestamp - origin) / NS, 'raw': r}
            for k in ('name', 'nameId'):
                if k in r:
                    e['name'] = strings.get(r[k], r[k])
            memory['events'].append(e)
    memory['events'].sort(key=lambda e: e['time_s'])
    # Decode operation labels dynamically, never assume numeric alloc/free codes.
    op_labels = {}
    for t, values in enums.items():
        if 'DEV_MEM_EVENT_OPER' in t or 'MEMORY_OPER' in t:
            for r in values:
                op_labels[r.get('id')] = str(r.get('label', r.get('name', '')))
    delta = collections.defaultdict(int)
    for e in memory['events']:
        r = e['raw']
        label = op_labels.get(r.get('memoryOperationType', r.get('operationType')), '').lower()
        device = (r.get('deviceId'), r.get('memKind'))
        size = r.get('bytes')
        if size is None or not label:
            continue
        change = -size if 'free' in label or 'release' in label or 'dealloc' in label else size if 'alloc' in label else None
        if change is not None:
            delta[device] += change
            memory['curve'].append({'time_s': e['time_s'], 'deviceId': device[0], 'memKind': device[1],
                                    'operation': label, 'change_bytes': change, 'observed_delta_bytes': delta[device]})
    driver_calls = []
    if 'CUPTI_ACTIVITY_KIND_DRIVER' in schema:
        for row in con.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_DRIVER WHERE start >= ? AND start <= ?', (origin, stop)):
            r = dict(row)
            label = str(strings.get(r.get('nameId'), ''))
            if label.startswith(('cuMemAlloc', 'cuMemFree', 'cuMemCreate', 'cuMemRelease', 'cuMemMap', 'cuMemUnmap')):
                driver_calls.append({'name': label, 'start_s': (r['start'] - origin) / NS,
                                     'end_s': (r['end'] - origin) / NS})
    memory['driver_api_timeline'] = driver_calls
    if not memory['curve']:
        report['warnings'].append('No decodable allocation byte events; cuMem API durations alone cannot reconstruct allocation sizes')


def add_metrics(report, con, schema, origin, stop, csv_path):
    required = {'sm_active': r'sms? active', 'tensor': r'tensor', 'dram_read': r'dram.*read',
                'dram_write': r'dram.*writ', 'gpc_clock': r'gpc.*clock|gpc.*frequency',
                'pcie_rx': r'pcie.*(rx|receive).*throughput|pcie.*read.*throughput', 'pcie_tx': r'pcie.*(tx|transmit).*throughput|pcie.*write.*throughput'}
    metrics = {'raw_csv': str(csv_path.resolve()), 'raw_source': 'SQLite contains all exported fields and samples',
               'required_fields': {}, 'series': [], 'downsampled': True, 'bin_width_s': .05,
               'statistics_source': 'Exact raw trailing sample intervals; bins are visualization only',
               'interval_convention': '(previous timestamp, timestamp], first sample has no inferred coverage'}
    report['gpu_metrics'] = metrics
    if 'GPU_METRICS' not in schema:
        report['warnings'].append('GPU_METRICS absent; hardware metrics unavailable')
        metrics['required_fields'] = {k: [] for k in required}
        return
    metadata = [dict(r) for r in con.execute('SELECT * FROM TARGET_INFO_GPU_METRICS')] if 'TARGET_INFO_GPU_METRICS' in schema else []
    selected = {}
    for label, pattern in required.items():
        matches = [m for m in metadata if re.search(pattern, m['metricName'], re.I)]
        metrics['required_fields'][label] = [{'typeId': m.get('typeId'), 'metricId': m['metricId'], 'name': m['metricName']} for m in matches]
        for m in matches:
            selected[(m.get('typeId'), m['metricId'])] = m
        if not matches:
            report['warnings'].append(f'Requested GPU metric unavailable: {label}')
    if not selected:
        return
    fields = schema['GPU_METRICS']
    clauses, params = [], []
    for type_id, metric_id in selected:
        if 'typeId' in fields and type_id is not None:
            clauses.append('(typeId=? AND metricId=?)')
            params.extend((type_id, metric_id))
        else:
            clauses.append('(metricId=?)')
            params.append(metric_id)
    groups = {key: {'timestamps': array('q'), 'values': array('d')} for key in selected}
    previous = {}
    raw_count = 0
    # Compressed directly: no second uncompressed multi-GB artifact.
    opener = gzip.open if str(csv_path).endswith('.gz') else open
    kwargs = {'compresslevel': 1} if str(csv_path).endswith('.gz') else {}
    with opener(csv_path, 'wt', newline='', **kwargs) as f:
        writer = csv.writer(f)
        writer.writerow(fields + ['metricName', 'start_s', 'end_s'])
        query = 'SELECT * FROM GPU_METRICS WHERE ' + ' OR '.join(clauses) + ' ORDER BY timestamp'
        for row in con.execute(query, params):
            r = dict(row)
            key = (r.get('typeId'), r['metricId'])
            t = r['timestamp']
            prev = previous.get(key)
            previous[key] = t
            writer.writerow([r.get(c) for c in fields] + [selected[key]['metricName'],
                            (prev - origin) / NS if prev is not None else '', (t - origin) / NS])
            raw_count += 1
            if t < origin or (prev is not None and prev >= stop):
                continue
            g = groups[key]
            if not g['timestamps'] and prev is not None:
                g['timestamps'].append(prev)
                g['values'].append(float('nan'))
            g['timestamps'].append(t)
            g['values'].append(float(r['value']) if r['value'] is not None else float('nan'))
    metrics['raw_csv_sample_count'] = raw_count
    print(f'Saved {raw_count} required raw metric rows; computing exact statistics and 50ms bins', flush=True)
    bin_ns = 50_000_000
    for key, g in groups.items():
        ts, vs = g['timestamps'], g['values']
        meta = selected[key]
        s = {'typeId': key[0], 'metricId': key[1], 'name': meta['metricName'], 'metadata': meta,
             'downsampled': True, 'bin_width_s': .05, 'raw_sample_count': len(ts),
             'columns': ['start_s', 'end_s', 'mean', 'min', 'max', 'coverage_s'], 'intervals': []}
        def weighted(windows):
            windows = union(windows)
            total = sum(b - a for a, b in windows)
            weighted_values = []
            covered, weighted_sum = 0, 0.
            for a, b in windows:
                left = max(1, bisect.bisect_right(ts, a))
                right = min(len(ts), bisect.bisect_left(ts, b) + 1)
                for i in range(left, right):
                    v = vs[i]
                    w = min(ts[i], b, stop) - max(ts[i-1], a, origin)
                    if w > 0 and math.isfinite(v):
                        weighted_values.append((v, w))
                        covered += w
                        weighted_sum += v * w
            p95, acc = None, 0
            weighted_values.sort()
            for v, w in weighted_values:
                acc += w
                if acc >= .95 * covered:
                    p95 = v
                    break
            return {'window_s': total / NS, 'coverage_s': covered / NS,
                    'coverage_fraction': covered / total if total else None,
                    'interval_count': len(weighted_values),
                    'mean': weighted_sum / covered if covered else None, 'p95': p95,
                    'max': weighted_values[-1][0] if weighted_values else None}
        s['measured_statistics'] = weighted([(origin, stop)])
        s['phase_statistics'] = []
        for p in report['phases']:
            def ns(t):
                return round(t * NS) + origin
            projected = [(ns(p['gpu_projected_start_s']), ns(p['gpu_projected_end_s']))] if p['gpu_projected_start_s'] is not None else []
            s['phase_statistics'].append({'phase_id': p['id'], 'name': p['name'],
                'cpu_window': weighted([(ns(p['cpu_start_s']), ns(p['cpu_end_s']))]),
                'gpu_projected_window': weighted(projected),
                'gpu_activity_union': weighted([(ns(a), ns(b)) for a, b in p['gpu_activity_union_intervals_s']])})
        bins = [[0., 0, math.inf, -math.inf] for _ in range(math.ceil((stop-origin)/bin_ns))]
        for i in range(1, len(ts)):
            a, b, v = max(origin, ts[i-1]), min(stop, ts[i]), vs[i]
            if b <= a or not math.isfinite(v):
                continue
            while a < b:
                j = (a-origin)//bin_ns
                edge = min(b, origin+(j+1)*bin_ns)
                w = edge-a
                z = bins[j]
                z[0] += v*w
                z[1] += w
                z[2] = min(z[2], v)
                z[3] = max(z[3], v)
                a = edge
        for j, (value_sum, coverage, lo, hi) in enumerate(bins):
            s['intervals'].append([j*bin_ns/NS, min((j+1)*bin_ns, stop-origin)/NS,
                value_sum/coverage if coverage else None, lo if coverage else None,
                hi if coverage else None, coverage/NS])
        metrics['series'].append(s)
        print(f"Finished metric {s['name']}", flush=True)


def annotate_metric_units(report):
    for series in report['gpu_metrics']['series']:
        clock = 'clock frequency' in series['name'].lower()
        series['raw_unit'] = 'Hz' if clock else '%'
        series['display_unit'] = 'MHz' if clock else '%'
        series['display_multiplier'] = 1e-6 if clock else 1
        if clock:
            series['unit_evidence'] = {
                'metric': 'gpc__cycles_elapsed.avg.per_second',
                'reference': 'https://docs.nvidia.com/nsight-systems/UserGuide/#gpu-metrics',
                'verified_capture_config': '/opt/nvidia/nsight-systems-cli/2026.4.1/target-linux-x64/GpuMetrics/gb20x.config',
                'config': 'GPC Clock Frequency: subMetrics name MHz, suffix .avg.per_second, multiplier 1.0e-06',
                'note': 'SQLite values and all analyzer statistics/bins retain Hz despite metricName MHz suffix'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', nargs='?', type=Path, default=Path(__file__).with_name('measured.sqlite'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('systems.json'))
    parser.add_argument('--metrics-csv', type=Path, default=Path(__file__).with_name('gpu_metrics_raw.csv.gz'))
    args = parser.parse_args()
    if not args.database.is_file():
        parser.exit(2, f'Not ready: {args.database}\n')
    analyze(args.database, args.output, args.metrics_csv)


if __name__ == '__main__':
    main()
