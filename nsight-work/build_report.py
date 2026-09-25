#!/usr/bin/env python3
"""Build an offline, measured-only Nsight HTML report (Python standard library).

Usage: python build_report.py analysis.json report-dir/index.html
Only the output HTML is written. Put measured.mp4 beside it; source artifact URLs
are used as provided (relative paths are relative to the output HTML).

Schema v1: title, arbitrary model/workload/hardware/software objects; sources
[{id,tool,artifact,version?,notes?}]; runs [{id,kind:'measured',source_ids,profiler,
duration_s,timing,phases,series,kernels,operators}]; ncu, capacity, findings.
Timing: {id,label,value_s,basis,source_id}. Phase: {id,label,start_s,end_s,basis,
parent_id?,source_id,gpu_start_s?,gpu_end_s?,gpu_busy_s?}.
Series: {id,label,panel,unit,scope,source_id,metric_name,status,reason?,sampling,
samples:[[start_s,end_s,value|null],...]}. sampling='interval' or 'instant'.
Panels: ram,vram,sm,tensor,dram,clock,pcie,cpu,power,temp,disk,swap. Intervals must not overlap.
Kernels: {id,name,family?,phase_id?,calls,total_gpu_s,min_s?,max_s?,source_id}.
Operators: {id,name,phase_id?,calls,cpu_total_s?,gpu_total_s?,accounting,source_id}.
NCU: {id,run_id,kernel_id?,kernel_name,launch?,phase_id?,source_id,representative,
counters:[{name,label?,value,unit,status,reason?}],notes?}.
Capacity: {id,label,resource,run_id?,phase_id?,observed,ceiling,unit,basis,
evidence_ids,assumptions?,interpretation?}. Numeric observed/ceiling or null.
Findings: {title,text,evidence_ids,kind:'observation'|'inference'}.
Optional run.phase_summaries: [{id,phase_id?,label,value,unit,basis,source_id}]
for supplied PyTorch allocated/reserved peaks or other measured summaries. No
allocator curves are generated from peaks. Capacity may include recommended
(numeric, same unit); observed, hardware ceiling, and recommendation stay separate.
Optional series.statistics={mean,p95,peak,coverage_pct} and
series.phase_statistics={phase_id:{mean,p95,peak,coverage_pct}} supply exact
full-raw statistics (values use the series unit). Overrides are authoritative,
including nulls; missing fields remain null. Their presence identifies already
binned chart input (50 ms); samples are preserved without rebinning. Missing
phase/window overrides on such series stay unavailable, never inferred from bin
means. Optional raw_sample_count preserves the analyzer's original sample count.
Without overrides, raw intervals determine statistics before 50 ms chart binning.
Instant samples remain unchanged; their sampled peak uses non-null timestamps
inside the closed window [start,end]. Mean, p95 and coverage stay null unless
explicitly supplied. An empty window has peak=null. Adjacent phase boundaries
may share an instantaneous observation.
Optional series.summary_role selects compact report columns explicitly:
sm,tensor,dram_read,dram_write,pcie_rx,pcie_tx,clock,rss,pss,private,system_used,device.
Absent roles are matched by panel and metric labels; unmatched columns stay null.
Optional phase.summary=true/false overrides main-table phase selection; otherwise
up to seven major phases and four steps are shown, with other phases expandable.
Optional series.phase_domain='host'|'gpu': defaults host for ram/cpu/disk/swap,
gpu for other panels. Host RAM uses host start_s/end_s; GPU uses projected bounds
when supplied, otherwise host bounds. Phase overrides must match that domain.
No non-measured records are accepted. Missing metrics stay missing; no timing or
hardware capacities are inferred. Phase statistics use the series phase_domain. Instant samples have no implicit
duration weights; only actual sampled peaks are computed.
"""
import argparse
import copy
import html
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

PANELS = ('ram', 'vram', 'sm', 'tensor', 'dram', 'clock', 'pcie', 'cpu', 'power', 'temp', 'disk', 'swap')


def number(value, label, nonnegative=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label}: expected finite number')
    if nonnegative and value < 0:
        raise ValueError(f'{label}: must be nonnegative')
    return value


def validate(data):
    if data.get('schema_version') != 1:
        raise ValueError('schema_version must be 1')
    # Never embed arbitrary archived reports or unvalidated extra fields.
    allowed = ('schema_version', 'title', 'model', 'workload', 'hardware', 'software',
               'sources', 'runs', 'ncu', 'capacity', 'findings')
    data = copy.deepcopy({key: data[key] for key in allowed if key in data})
    def guard(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                guard(key); guard(value)
        elif isinstance(obj, list):
            for item in obj: guard(item)
        elif isinstance(obj, str) and 'warmup' in obj.lower().replace('-', '').replace('_', '').replace(' ', ''):
            raise ValueError('Input contains excluded pre-measurement content; provide measured-only analysis')
        elif isinstance(obj, float) and not math.isfinite(obj):
            raise ValueError('Non-finite JSON number')
    guard(data)
    for key in ('model', 'workload', 'hardware', 'software'):
        if not isinstance(data.setdefault(key, {}), dict): raise ValueError(f'{key} must be an object')
    for key in ('sources', 'runs', 'ncu', 'capacity', 'findings'):
        if not isinstance(data.setdefault(key, []), list): raise ValueError(f'{key} must be an array')
    def unique(rows, name):
        ids = [r['id'] for r in rows]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError(f'{name}: IDs must be unique nonempty strings')
    unique(data['sources'], 'sources'); unique(data['runs'], 'runs')
    sources = {s['id'] for s in data['sources']}
    def source(row):
        if row.get('source_id') not in sources: raise ValueError(f'Unknown source_id: {row.get("source_id")}')
    for s in data['sources']:
        artifact = s.get('artifact', '')
        if not isinstance(artifact, str): raise ValueError('artifact must be a URL/path string')
        parsed = urlsplit(artifact)
        if parsed.scheme not in ('', 'http', 'https') or artifact.startswith('//') or '\\' in artifact:
            raise ValueError('artifact must be a relative path or HTTP(S) URL')
    if not data['runs']: raise ValueError('At least one measured run is required')
    for run in data['runs']:
        if run.get('kind') != 'measured': raise ValueError('Every run must have kind="measured"')
        duration = number(run['duration_s'], 'duration_s')
        if duration <= 0: raise ValueError('duration_s must be positive')
        for sid in run.get('source_ids', []):
            if sid not in sources: raise ValueError(f'Unknown source: {sid}')
        for key in ('timing', 'phases', 'series', 'kernels', 'operators', 'phase_summaries'):
            run.setdefault(key, [])
            unique(run[key], key)
            for row in run[key]: source(row)
        phase_ids = {p['id'] for p in run['phases']}
        def bounds(start, end, label):
            number(start, label); number(end, label)
            if not 0 <= start <= end <= duration: raise ValueError(f'{label}: outside measured window or reversed')
        for phase in run['phases']:
            bounds(phase['start_s'], phase['end_s'], 'phase')
            if phase.get('parent_id') is not None and phase['parent_id'] not in phase_ids: raise ValueError('Unknown parent phase')
            if 'gpu_start_s' in phase or 'gpu_end_s' in phase:
                bounds(phase['gpu_start_s'], phase['gpu_end_s'], 'GPU phase')
            if 'gpu_busy_s' in phase:
                busy = number(phase['gpu_busy_s'], 'gpu_busy_s')
                span = phase.get('gpu_end_s', phase['end_s']) - phase.get('gpu_start_s', phase['start_s'])
                if busy > span + 1e-9: raise ValueError('GPU busy union exceeds phase span')
        for timing in run['timing']: number(timing['value_s'], 'timing value_s')
        for series in run['series']:
            if series['panel'] not in PANELS: raise ValueError('Unknown series panel')
            if series['sampling'] not in ('interval', 'instant'): raise ValueError('Unknown sampling mode')
            if 'raw_sample_count' in series:
                count = number(series['raw_sample_count'], 'raw_sample_count')
                if int(count) != count: raise ValueError('raw_sample_count must be an integer')
            overrides = series.get('phase_statistics', {})
            if not isinstance(overrides, dict): raise ValueError('phase_statistics must be an object')
            if not set(overrides).issubset(phase_ids): raise ValueError('phase_statistics has unknown phase_id')
            records = list(overrides.values())
            if 'statistics' in series: records.append(series['statistics'])
            for stats in records:
                if not isinstance(stats, dict): raise ValueError('statistics must be an object')
                for key in ('mean', 'p95', 'peak', 'coverage_pct'):
                    if stats.get(key) is not None:
                        number(stats[key], key, key == 'coverage_pct')
                        if key == 'coverage_pct' and stats[key] > 100: raise ValueError('coverage_pct exceeds 100')
            if series.get('phase_domain', 'host' if series['panel'] in ('ram', 'cpu', 'disk', 'swap') else 'gpu') not in ('host', 'gpu'):
                raise ValueError('phase_domain must be host or gpu')
            previous = -1
            for sample in series.get('samples', []):
                if len(sample) != 3: raise ValueError('Samples must be [start_s,end_s,value]')
                a, b, v = sample; bounds(a, b, 'sample')
                if a < previous: raise ValueError('Samples must be sorted and nonoverlapping')
                if series['sampling'] == 'instant' and a != b: raise ValueError('Instant sample must have equal timestamps')
                if series['sampling'] == 'interval' and a == b: raise ValueError('Interval samples need positive duration')
                if v is not None: number(v, 'sample value', False)
                previous = b
        for row in run['phase_summaries']:
            if row.get('phase_id') is not None and row['phase_id'] not in phase_ids: raise ValueError('Unknown summary phase_id')
            if row.get('value') is not None: number(row['value'], 'phase summary value')
        for row in run['kernels'] + run['operators']:
            if row.get('phase_id') is not None and row['phase_id'] not in phase_ids: raise ValueError('Unknown phase_id')
            calls = number(row['calls'], 'calls')
            if int(calls) != calls: raise ValueError('calls must be an integer')
            for key in ('total_gpu_s', 'min_s', 'max_s', 'cpu_total_s', 'gpu_total_s'):
                if row.get(key) is not None: number(row[key], key)
    runs = {r['id']: r for r in data['runs']}
    for ncu in data['ncu']:
        source(ncu)
        if ncu['run_id'] not in runs: raise ValueError('NCU has unknown run_id')
        for counter in ncu.get('counters', []):
            if counter.get('value') is not None: number(counter['value'], 'NCU counter', False)
    for row in data['capacity']:
        for key in ('observed', 'ceiling', 'recommended'):
            if row.get(key) is not None: number(row[key], key)
        if row.get('run_id') is not None and row['run_id'] not in runs: raise ValueError('Capacity has unknown run_id')
    return data


def weighted(series, start, end):
    if series['sampling'] == 'instant':
        values = [v for a, _, v in series.get('samples', [])
                  if v is not None and start <= a <= end]
        return dict(mean=None, p95=None, peak=max(values) if values else None,
                    coverage_pct=None)
    points = [(v, max(0, min(b, end) - max(a, start)))
              for a, b, v in series.get('samples', []) if v is not None and b > start and a < end]
    points = [(v, w) for v, w in points if w > 0]
    covered = sum(w for _, w in points)
    if not covered: return None
    cumulative = 0
    for v, w in sorted(points):
        cumulative += w
        if cumulative >= covered * .95:
            p95 = v; break
    return dict(mean=sum(v*w for v, w in points)/covered, p95=p95, peak=max(v for v, _ in points),
                coverage_pct=100*covered/(end-start) if end > start else 0)


def bin_samples(series, duration, width=.05):
    """50 ms chart means; gaps and null values contribute no coverage."""
    buckets = {}
    for a, b, value in series.get('samples', []):
        if value is None: continue
        if series['sampling'] == 'instant':
            key = int(a / width)
            entry = buckets.setdefault(key, [0., 0.])
            entry[0] += value; entry[1] += 1
        else:
            while a < b:
                key = int(a / width)
                edge = min(b, (key + 1) * width)
                if edge <= a: key += 1; edge = min(b, (key + 1) * width)
                weight = edge - a
                entry = buckets.setdefault(key, [0., 0.])
                entry[0] += value * weight; entry[1] += weight
                a = edge
    output = []
    for key, (total, weight) in sorted(buckets.items()):
        start, end = key * width, min(duration, (key + 1) * width)
        if series['sampling'] == 'instant': end = start
        output.append([start, end, total / weight])
    return output


def statistics(series, start, end, phase_id=None):
    """Never estimate raw quantiles or peaks from supplied chart means."""
    supplied = series.get('statistics') if phase_id is None else series.get('phase_statistics', {}).get(phase_id)
    if supplied is not None:
        return {key: supplied.get(key) for key in ('mean', 'p95', 'peak', 'coverage_pct')}
    if series['sampling'] == 'interval' and ('statistics' in series or 'phase_statistics' in series):
        return None
    return weighted(series, start, end)


def phase_window(series, phase):
    domain = series.get('phase_domain', 'host' if series['panel'] in ('ram', 'cpu', 'disk', 'swap') else 'gpu')
    if domain == 'gpu' and 'gpu_start_s' in phase:
        return phase['gpu_start_s'], phase['gpu_end_s'], 'GPU-projected'
    return phase['start_s'], phase['end_s'], 'Host'


def derive(data):
    for run in data['runs']:
        families = {}
        for kernel in run['kernels']:
            name = kernel.get('family') or 'Unclassified'
            family = families.setdefault(name, dict(name=name, calls=0, total_gpu_s=0))
            family['calls'] += kernel['calls']; family['total_gpu_s'] += kernel['total_gpu_s']
        total = sum(f['total_gpu_s'] for f in families.values())
        run['families'] = sorted(families.values(), key=lambda f: -f['total_gpu_s'])
        for f in run['families']: f['share_pct'] = 100*f['total_gpu_s']/total if total else None
        for p in run['phases']:
            p['weighted'] = {}
            p['statistics_windows'] = {}
            for series in run['series']:
                start, end, domain = phase_window(series, p)
                p['weighted'][series['id']] = statistics(series, start, end, p['id'])
                p['statistics_windows'][series['id']] = dict(start_s=start, end_s=end, domain=domain)
        run['raw_summary'] = {s['id']: statistics(s, 0, run['duration_s']) for s in run['series']}
        for series in run['series']:
            pre_binned = series['sampling'] == 'interval' and ('statistics' in series or 'phase_statistics' in series)
            series['input_sample_count'] = len(series.get('samples', []))
            series.setdefault('raw_sample_count', None if pre_binned else series['input_sample_count'])
            series['statistics_basis'] = ('Exact full-raw overrides; 50 ms pre-binned chart source'
                                          if pre_binned else 'Computed from supplied raw intervals'
                                          if series['sampling'] == 'interval' else 'Original instantaneous samples; no duration weights')
            if series['sampling'] == 'interval':
                if not pre_binned: series['samples'] = bin_samples(series, run['duration_s'])
                series['chart_bin_s'] = .05
    return data


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; media-src 'self' file:; img-src 'self' file: data:; base-uri 'none'; object-src 'none'">
<title>__TITLE__</title><style>
:root{color-scheme:light dark;--bg:#f6f4ef;--fg:#181817;--muted:#68655f;--panel:#fffefa;--line:#d8d4ca;--blue:#2c67c9}
@media(prefers-color-scheme:dark){:root{--bg:#151513;--fg:#f2efe8;--muted:#aaa69e;--panel:#201f1b;--line:#39372f;--blue:#73a7ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:32px 24px 70px}h1{font-size:clamp(28px,4vw,48px);line-height:1.1;letter-spacing:-.04em;margin:8px 0 18px}h2{font-size:23px;margin:32px 0 12px}h3{margin:12px 0}.kicker{color:var(--blue);letter-spacing:.1em;text-transform:uppercase;font-size:12px}.note,small{color:var(--muted)}.panel,.stat{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:18px;min-width:0}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:18px 0}.stat strong{display:block;font-size:27px}.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:12px 0}button,select,input{font:inherit;background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:7px}button{cursor:pointer}button:hover{border-color:var(--blue)}input[type=number]{width:100px}a{color:var(--blue)}.scroll{overflow:auto;max-height:540px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{position:sticky;top:0;background:var(--panel);font-size:12px;color:var(--muted);z-index:1}td{max-width:600px;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}details{margin:12px 0}summary{cursor:pointer}.chart{display:block;width:100%;height:166px;touch-action:none;user-select:none}.chart text{fill:var(--muted);font:11px system-ui}.chart .grid{stroke:var(--line)}.chart .cursor{stroke:var(--fg);stroke-dasharray:3 3}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px}.phase-buttons{display:flex;gap:6px;flex-wrap:wrap}.phase-buttons button{font-size:12px}.chart-panel{margin:14px 0;border-top:1px solid var(--line);padding-top:8px}.hover{min-height:24px;color:var(--muted);font-variant-numeric:tabular-nums}.badge{border:1px solid var(--line);padding:3px 7px;border-radius:12px}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:16px}video{width:100%;max-height:500px;background:#111}.empty{padding:16px;color:var(--muted)}footer{margin-top:30px;color:var(--muted)}@media(max-width:700px){main{padding:20px 12px}.twocol{grid-template-columns:1fr}}@media print{.controls,video{display:none}.scroll{max-height:none;overflow:visible}.panel{break-inside:avoid}}
.findings{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin:18px 0}.recommendation{border-left:4px solid var(--blue);margin:12px 0}.recommendation strong{font-size:23px}.exact-name{max-width:360px;margin:0}.exact-name summary{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:340px}.exact-name pre{max-width:520px}.compact td,.compact th{padding:8px 6px;font-size:12px}.compact th{max-width:100px}.metric-note{font-size:11px;color:var(--muted)}details>.panel{margin-top:12px}</style></head><body><main><header><div class="kicker">Nsight Systems + Nsight Compute · measured captures</div><h1 id="title"></h1><p class="note">Measured timing, synchronized telemetry, kernel activity and capacity evidence.</p></header>
<div class="controls"><label>Measured run <select id="run"></select></label><span id="profiler" class="badge"></span></div><div id="stats" class="stats"></div>
<div id="capacity-headlines" class="stats"></div><div id="recommendation"></div><div id="findings" class="findings"></div><details><summary>Additional findings and measurement limitations</summary><div id="more-findings" class="findings"></div></details><details><summary>Exact model and workload</summary><div class="twocol"><div class="panel" id="model"></div><div class="panel" id="workload"></div></div></details>
<details><summary>Hardware and software</summary><div class="twocol"><div id="hardware"></div><div id="software"></div></div></details>
<h2>Synchronized resource timeline</h2><div class="panel"><div class="controls"><button id="reset">Reset</button><button id="zin">Zoom in</button><button id="zout">Zoom out</button><button id="left">← Pan</button><button id="right">Pan →</button><label>Start s <input id="start" type="number" min="0" step="any"></label><label>End s <input id="end" type="number" min="0" step="any"></label><button id="apply">Apply range</button><span id="range" aria-live="polite"></span></div><p class="note">Drag to select a time range; wheel to zoom. All charts share one range and cursor. Click a phase to zoom. Charts use 50 ms means; gaps smaller than one bin may be hidden. Summaries use exact analyzer overrides where supplied, otherwise raw intervals. Missing overrides for pre-binned sources remain unavailable. Instantaneous samples are points.</p><div id="phases" class="phase-buttons"></div><div id="charts"></div></div>
<details><summary>All resource statistics and measurement provenance</summary><div id="resource-summary"></div></details><h2>Phase measurements</h2><p class="note">Compact view: major phases and denoising steps. Coverage is the min–max across available displayed GPU metrics; missing counters remain —. Host RAM uses host bounds; GPU metrics use GPU-projected bounds when supplied. Analyzer overrides provide exact full-raw statistics for that domain. Otherwise means and p95 are duration-weighted over supplied raw intervals. Coverage excludes missing samples. Nested phase times are not additive. Instantaneous samples report only actual sampled peaks. — means unavailable; no weights, mean, p95 or coverage are invented.</p><div id="phase-table"></div><details><summary>Loader and other phase details</summary><div id="phase-detail"></div></details><details><summary>Exact phase bounds, GPU busy times and all metric statistics</summary><div id="phase-exact"></div></details>
<h3>Per-phase memory peaks and supplied summaries</h3><p class="note">One row per phase. RAM uses host phase bounds; device memory follows its configured phase domain. Peaks use exact analyzer overrides or actual samples, including instantaneous telemetry. Values are GiB. Supplied allocator peaks are scalar measurements; no allocator curve is inferred.</p><div id="memory-peaks"></div><details><summary>Other phase memory peaks and supplied allocator measurements</summary><div id="memory-detail"></div><div id="phase-summaries"></div></details><h2>Kernel families</h2><p class="note">Derived from supplied kernel rows. Shares use summed kernel durations, which may overlap and are not elapsed-time utilization.</p><div id="families"></div>
<h2>Kernels and operators</h2><div class="controls"><label>Filter names <input id="filter" type="search" placeholder="Kernel or operator name"></label></div><h3>Dominant kernels</h3><div id="kernels"></div><div id="kernel-pager" class="controls"></div><details><summary>Operators · browse and filter</summary><p class="note">CPU/GPU totals preserve supplied inclusive or exclusive accounting.</p><div id="operators"></div><div id="operator-pager" class="controls"></div></details>
<h2>Nsight Compute counters</h2><p class="note">Counter collection and replay perturb execution. NCU durations are never used as whole-run benchmark timings. Representative launches do not establish whole-run utilization.</p><div id="ncu"></div>
<h2>Capacity analysis</h2><p class="note">Headroom is ceiling minus observed value in matching units. It is not a throughput or concurrency prediction. The basis, evidence and assumptions accompany each comparison.</p><details><summary>Exact capacity comparisons and evidence</summary><div id="capacity"></div></details>
<h2>Measured output</h2><div class="panel"><video controls preload="metadata" poster="assets/poster.jpg"><source src="measured.mp4" type="video/mp4"><source src="assets/preview.webm" type="video/webm"></video><p class="note"><a href="measured.mp4" download>Download the measured MP4</a> · 720 × 1280 · 241 frames · 24 FPS · native generated audio. A WebM browser fallback is encoded from the same video after profiling; preview conversion is excluded from all timings.</p></div>
<h2>Sources and downloads</h2><details><summary>Source artifacts, downloads and collection notes</summary><div id="sources"></div></details><details><summary>Input schema and measurement conventions</summary><pre>__SCHEMA__</pre></details><footer>Offline report · inline SVG and JavaScript · no external libraries or network requests required</footer></main>
<script id="data" type="application/json">__DATA__</script><script>
'use strict';
const D=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id), esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(v,n=3)=>v===null||v===undefined?'—':typeof v==='number'?v.toLocaleString(undefined,{maximumFractionDigits:n}):String(v);
const colors=['#2c79d9','#00a5a9','#bf850e','#9562cf','#d45b5b','#32966c'];
const display=(v,u)=>v==null?'—':fmt(u==='bytes'?v/2**30:v), unit=u=>u==='bytes'?'GiB':u;
const text=v=>typeof v==='object'?JSON.stringify(v):String(v??'—');
function table(id,heads,rows){$(id).innerHTML=rows.length?'<div class="panel scroll"><table><thead><tr>'+heads.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(c=>'<td>'+c+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>':'<div class="panel empty">No measured data supplied.</div>';}
function metadata(id,obj){$(id).innerHTML='<h3>'+esc(id[0].toUpperCase()+id.slice(1))+'</h3>'+Object.entries(obj).map(([k,v])=>'<p><strong>'+esc(k)+'</strong><br>'+esc(text(v))+'</p>').join('');}
$('title').textContent=D.title||'H3 ref2va — Nsight benchmark';['model','workload','hardware','software'].forEach(k=>metadata(k,D[k]));
$('run').innerHTML=D.runs.map((r,i)=>'<option value="'+i+'">'+esc(r.id)+'</option>').join('');
const basePanels=['ram','vram','sm','tensor','dram','clock','pcie'];
const panels=()=>[...basePanels,...['cpu','power','temp','disk','swap'].filter(p=>R.series.some(s=>s.panel===p))];
let R,lo=0,hi=1,cursor=null,drag=null,kernelPage=0,operatorPage=0;
const bounds=p=>[p.gpu_start_s??p.start_s,p.gpu_end_s??p.end_s];
function setRange(a,b){if(!Number.isFinite(a)||!Number.isFinite(b)||b<=a)return;let width=Math.min(b-a,R.duration_s);width=Math.max(Math.min(.000001,R.duration_s),width);lo=Math.max(0,Math.min(a,R.duration_s-width));hi=lo+width;draw();}
function zoom(f,center=(lo+hi)/2){setRange(center-(center-lo)*f,center+(hi-center)*f);}
function render(){kernelPage=0;operatorPage=0;R=D.runs[+$('run').value];lo=0;hi=R.duration_s;cursor=null;$('profiler').textContent=R.profiler||'Profiler unspecified';
$('stats').innerHTML='<div class="stat"><small>Measured window</small><strong>'+fmt(R.duration_s)+' s</strong><small>'+esc(R.id)+'</small></div>'+R.timing.map(t=>'<div class="stat"><small>'+esc(t.label)+'</small><strong>'+esc(fmt(t.value_s))+' s</strong><small>'+esc(t.basis)+' · '+esc(t.source_id)+'</small></div>').join('');
$('phases').innerHTML=R.phases.map((p,i)=>'<button data-phase="'+i+'">'+esc(p.label)+'</button>').join('');$('phases').querySelectorAll('button').forEach(b=>b.onclick=()=>setRange(...bounds(R.phases[+b.dataset.phase])));
renderPhaseTables();
table('phase-summaries',['Phase','Measurement','Value','Basis / source'],(R.phase_summaries||[]).map(s=>[esc(s.phase_id??'Measured run'),esc(s.label),display(s.value,s.unit)+' '+esc(unit(s.unit)),esc(s.basis)+' · '+esc(s.source_id)]));
table('families',['Family','Kernel total s','Launches','Share %'],R.families.map(f=>[esc(f.name),fmt(f.total_gpu_s),fmt(f.calls,0),fmt(f.share_pct,2)]));
table('resource-summary',['Series','Mean','p95','Peak','Coverage %','Raw samples','Input samples','Statistics basis'],R.series.map(s=>{const w=R.raw_summary[s.id];return [esc(s.label)+' ('+esc(unit(s.unit))+')',w?display(w.mean,s.unit):'—',w?display(w.p95,s.unit):'—',w?display(w.peak,s.unit):'—',w?fmt(w.coverage_pct,1):'—',fmt(s.raw_sample_count,0),fmt(s.input_sample_count,0),esc(s.statistics_basis)];}));
renderRows();draw();
const ncu=D.ncu.filter(n=>n.run_id===R.id);$('ncu').innerHTML=ncu.map((n,i)=>'<details class="panel"><summary>'+esc(n.kernel_name.slice(0,110))+' · '+(n.counters||[]).length+' counters</summary><pre>'+esc(n.kernel_name)+'</pre><p class="note">'+esc(n.id)+' · '+esc(n.source_id)+' · '+(n.representative?'Representative launch':'Launch collection')+' · '+esc(text(n.launch))+'</p><div id="counter-'+i+'"></div><p>'+esc(text(n.notes??''))+'</p></details>').join('')||'<p class="empty">No NCU counters supplied.</p>';ncu.forEach((n,i)=>table('counter-'+i,['Counter','Value','Unit','Status / reason'],(n.counters||[]).map(c=>[esc(c.name)+(c.label?'<br>'+esc(c.label):''),esc(fmt(c.value)),esc(c.unit),esc((c.status||'unspecified')+' '+(c.reason||''))])));
const caps=D.capacity.filter(c=>!c.run_id||c.run_id===R.id);renderHeadlines(caps);table('capacity',['Resource','Observed','Hardware ceiling','Recommended','Headroom','Basis / evidence','Assumptions / interpretation'],caps.map(c=>[esc(c.label||c.resource),display(c.observed,c.unit)+' '+esc(unit(c.unit)),display(c.ceiling,c.unit)+' '+esc(unit(c.unit)),display(c.recommended,c.unit)+' '+esc(unit(c.unit)),c.observed!=null&&c.ceiling!=null?display(c.ceiling-c.observed,c.unit)+' '+esc(unit(c.unit)):'—',esc(c.basis)+'<br>'+esc((c.evidence_ids||[]).join(', ')),esc(text(c.assumptions??''))+'<br>'+esc(c.interpretation??'')]));}

function longName(name){return '<details class="exact-name"><summary title="'+esc(name)+'">'+esc(name)+'</summary><pre>'+esc(name)+'</pre></details>';}
function seriesRole(role,panel,pattern){const explicit=R.series.find(s=>s.summary_role===role);if(explicit)return explicit;return R.series.find(s=>s.panel===panel&&(!pattern||pattern.test([s.id,s.label,s.metric_name,s.scope].join(' '))));}
function selectedMetrics(){return [
 ['SM',seriesRole('sm','sm'),true],['Tensor',seriesRole('tensor','tensor')],
 ['DRAM read',seriesRole('dram_read','dram',/read|\brd\b/i)],['DRAM write',seriesRole('dram_write','dram',/write|\bwr\b/i)],
 ['PCIe RX',seriesRole('pcie_rx','pcie',/\brx\b|receive|pcie_rx/i)],['PCIe TX',seriesRole('pcie_tx','pcie',/\btx\b|transmit|pcie_tx/i)],
 ['Clock',seriesRole('clock','clock')]];}
function memoryMetrics(){return [['RSS',seriesRole('rss','ram',/rss|resident/i)],['PSS',seriesRole('pss','ram',/pss|proportional/i)],['Private',seriesRole('private','ram',/private/i)],['System used',seriesRole('system_used','ram',/system.*used|used.*system/i)],['Device',seriesRole('device','vram',/device|nvml|nvidia/i)]];}
function phaseGroups(){const steps=R.phases.filter(p=>/step[ _/-]*\d|denois.*\d/i.test(p.label+' '+p.id));const hasDenoising=R.phases.some(p=>/denoising_loop/i.test(p.label));const major=R.phases.filter(p=>!steps.includes(p)&&!(hasDenoising&&/SamplerCustomAdvanced/i.test(p.label))&&!(/load|setup|schedul|tokeniz/i.test(p.label+' '+p.id)&&(p.end_s-p.start_s)<R.duration_s*.01)).sort((a,b)=>(b.end_s-b.start_s)-(a.end_s-a.start_s)).slice(0,7);const main=R.phases.filter(p=>p.summary===true||(p.summary!==false&&(major.includes(p)||steps.slice(0,4).includes(p))));return [main,R.phases.filter(p=>!main.includes(p))];}
function renderPhaseTables(){const groups=phaseGroups(),metrics=selectedMetrics(),heads=['Phase','Host / GPU s',...metrics.map(([label,s,p95])=>label+' '+(p95?'mean / p95':'mean')+(s?' ('+unit(s.unit)+')':'')),'Coverage %'];
const rows=phases=>phases.map(p=>{const cov=metrics.map(([,s])=>s?p.weighted[s.id]?.coverage_pct:null).filter(v=>v!=null);return [esc(p.label),fmt(p.end_s-p.start_s)+' / '+(p.gpu_start_s==null?'—':fmt(p.gpu_end_s-p.gpu_start_s)),...metrics.map(([,s,p95])=>{const w=s?p.weighted[s.id]:null;return w?display(w.mean,s.unit)+(p95?' / '+display(w.p95,s.unit):''):'—';}),cov.length?fmt(Math.min(...cov),1)+(Math.max(...cov)!==Math.min(...cov)?'–'+fmt(Math.max(...cov),1):''):'—'];});
['phase-table','phase-detail'].forEach((id,i)=>{table(id,heads,rows(groups[i]));$(id).className='compact';});
$('phase-exact').innerHTML=R.phases.map(p=>'<details><summary>'+esc(p.label)+'</summary><pre>'+esc(JSON.stringify(p,null,2))+'</pre></details>').join('');
const mem=memoryMetrics();['memory-peaks','memory-detail'].forEach((id,i)=>{table(id,['Phase',...mem.map(([label])=>label+' peak GiB')],groups[i].map(p=>[esc(p.label),...mem.map(([,s])=>{const v=s?p.weighted[s.id]?.peak:null;return s?display(v,s.unit):'—';})]));$(id).className='compact';});
}
function renderHeadlines(caps){const mem=memoryMetrics().filter(([,s])=>s&&R.raw_summary[s.id]?.peak!=null);$('capacity-headlines').innerHTML=mem.map(([label,s])=>'<div class="stat"><small>Observed '+esc(label)+' peak</small><strong>'+display(R.raw_summary[s.id].peak,s.unit)+' '+esc(unit(s.unit))+'</strong><small>'+esc(s.label)+'</small></div>').join('');
$('recommendation').innerHTML=caps.filter(c=>c.recommended!=null).map(c=>'<article class="panel recommendation"><small>'+esc(c.label||c.resource)+' · recommended budget</small><br><strong>'+display(c.recommended,c.unit)+' '+esc(unit(c.unit))+'</strong><p>'+esc(c.interpretation||'')+'</p><p class="note">'+esc(text(c.assumptions??''))+'</p><small>Inference · '+esc(c.basis)+' · '+esc((c.evidence_ids||[]).join(', '))+'</small></article>').join('');}
function pager(id,page,count,change){const pages=Math.max(1,Math.ceil(count/20));$(id).innerHTML='<button id="'+id+'-prev" '+(page===0?'disabled':'')+'>Previous</button><span>'+fmt(count? page*20+1:0,0)+'–'+fmt(Math.min((page+1)*20,count),0)+' of '+fmt(count,0)+'</span><button id="'+id+'-next" '+(page+1>=pages?'disabled':'')+'>Show next 20</button>';$(id+'-prev').onclick=()=>change(Math.max(0,page-1));$(id+'-next').onclick=()=>change(Math.min(pages-1,page+1));}
function renderRows(){let q=$('filter').value.toLowerCase();const kernels=R.kernels.filter(k=>k.name.toLowerCase().includes(q)).slice().sort((a,b)=>b.total_gpu_s-a.total_gpu_s),operators=R.operators.filter(o=>o.name.toLowerCase().includes(q)).slice().sort((a,b)=>(b.gpu_total_s??b.cpu_total_s??0)-(a.gpu_total_s??a.cpu_total_s??0));kernelPage=Math.min(kernelPage,Math.max(0,Math.ceil(kernels.length/20)-1));operatorPage=Math.min(operatorPage,Math.max(0,Math.ceil(operators.length/20)-1));
table('kernels',['Kernel / family','GPU total s','Launches','Mean µs','Phase / source'],kernels.slice(kernelPage*20,kernelPage*20+20).map(k=>[longName(k.name)+'<small>'+esc(k.family||'Unclassified')+'</small>',fmt(k.total_gpu_s),fmt(k.calls,0),k.calls?fmt(k.total_gpu_s/k.calls*1e6):'—',esc(k.phase_id??'All supplied rows')+'<br>'+esc(k.source_id)]));
table('operators',['Operator','Calls','CPU s','GPU s','Accounting'],operators.slice(operatorPage*20,operatorPage*20+20).map(o=>[longName(o.name),fmt(o.calls,0),fmt(o.cpu_total_s),fmt(o.gpu_total_s),esc(o.accounting)]));pager('kernel-pager',kernelPage,kernels.length,n=>{kernelPage=n;renderRows();});pager('operator-pager',operatorPage,operators.length,n=>{operatorPage=n;renderRows();});}
function draw(){ $('start').value=lo.toFixed(6);$('end').value=hi.toFixed(6);$('range').textContent=fmt(lo)+'–'+fmt(hi)+' s';
$('charts').innerHTML=panels().map(panel=>'<div class="chart-panel"><strong>'+panel.toUpperCase()+'</strong><div class="legend" id="legend-'+panel+'"></div><svg class="chart" id="chart-'+panel+'" viewBox="0 0 1100 166" role="img" aria-label="'+panel+' timeline"></svg><div class="hover" id="hover-'+panel+'"></div></div>').join('');
for(const panel of panels()){const list=R.series.filter(s=>s.panel===panel),svg=$('chart-'+panel),x=t=>74+(t-lo)/(hi-lo)*1000;
$('legend-'+panel).innerHTML=list.map((s,i)=>'<span style="color:'+colors[i%colors.length]+'">● '+esc(s.label)+' ('+esc(unit(s.unit))+')</span>').join('')+(list.length?'<details><summary>Metric definitions and provenance</summary>'+list.map(s=>'<p>'+esc(s.label)+' · '+esc(s.metric_name)+' · '+esc(s.scope)+' · '+esc(s.source_id)+' · '+esc(s.status)+' · '+esc(s.statistics_basis)+(s.reason?' — '+esc(s.reason):'')+'</p>').join('')+'</details>':'Not collected');
let markup='';for(let i=0;i<=5;i++){let t=lo+(hi-lo)*i/5;markup+='<line class="grid" x1="'+x(t)+'" x2="'+x(t)+'" y1="12" y2="138"/><text x="'+x(t)+'" y="158" text-anchor="middle">'+fmt(t)+' s</text>';}
R.phases.forEach((p,i)=>{let[a,b]=bounds(p);a=Math.max(a,lo);b=Math.min(b,hi);if(b>a)markup+='<rect x="'+x(a)+'" y="12" width="'+(x(b)-x(a))+'" height="126" fill="'+colors[i%colors.length]+'" opacity=".055"><title>'+esc(p.label)+'</title></rect>';});
// Mixed units get independent labeled scales; no unit conversion between counters.
const groups=[...new Set(list.map(s=>s.unit))];let groupMax={};for(const u of groups){let max=0;for(const s of list.filter(s=>s.unit===u))for(const[a,b,v]of s.samples||[])if(v!=null&&b>=lo&&a<=hi)max=Math.max(max,u==='bytes'?v/2**30:v);groupMax[u]=max||1;}
list.forEach((s,i)=>{let max=groupMax[s.unit],y=v=>138-(s.unit==='bytes'?v/2**30:v)/max*116,color=colors[i%colors.length],segments=[];for(const[a,b,v]of s.samples||[]){if(v==null||b<lo||a>hi)continue;if(s.sampling==='instant'){segments.push('<circle cx="'+x(a)+'" cy="'+y(v)+'" r="2" fill="'+color+'"/>');}else segments.push('<path d="M'+x(Math.max(a,lo))+','+y(v)+'H'+x(Math.min(b,hi))+'" stroke="'+color+'" stroke-width="1.6" fill="none"/>');}markup+=segments.join('');});
groups.forEach((u,i)=>{markup+='<text x="4" y="'+(22+i*14)+'">'+fmt(groupMax[u],1)+' '+esc(unit(u))+'</text>';});markup+='<text x="45" y="138">0</text><line class="cursor" x1="0" x2="0" y1="12" y2="138" visibility="hidden"/>';
if(!list.length)markup+='<text x="550" y="80" text-anchor="middle">No measured samples supplied</text>';svg.innerHTML=markup;
const time=e=>lo+Math.max(0,Math.min(1,((e.clientX-svg.getBoundingClientRect().left)/svg.getBoundingClientRect().width*1100-74)/1000))*(hi-lo);
svg.onpointermove=e=>{cursor=time(e);updateCursor();};svg.onpointerleave=()=>{if(!drag){cursor=null;updateCursor();}};svg.onpointerdown=e=>{drag={t:time(e),x:e.clientX};svg.setPointerCapture(e.pointerId);};svg.onpointerup=e=>{if(drag){let start=drag;drag=null;let end=time(e);if(Math.abs(e.clientX-start.x)>4)setRange(Math.min(start.t,end),Math.max(start.t,end));}};svg.onpointercancel=()=>drag=null;svg.onwheel=e=>{e.preventDefault();zoom(e.deltaY>0?1.3:.77,time(e));};
}updateCursor();}
function updateCursor(){for(const panel of panels()){let line=$('chart-'+panel).querySelector('.cursor');line.setAttribute('visibility',cursor!==null&&cursor>=lo&&cursor<=hi?'visible':'hidden');if(cursor===null){$('hover-'+panel).textContent='';continue;}let x=74+(cursor-lo)/(hi-lo)*1000;line.setAttribute('x1',x);line.setAttribute('x2',x);$('hover-'+panel).textContent=fmt(cursor)+' s · '+R.series.filter(s=>s.panel===panel).map(s=>{let samples=s.samples||[],left=0,right=samples.length;while(left<right){let mid=(left+right)>>1;if(samples[mid][0]<=cursor)left=mid+1;else right=mid;}let p=samples[left-1];let value=p&&s.sampling==='interval'&&cursor<p[1]?p[2]:null;return s.label+': '+display(value,s.unit)+' '+unit(s.unit);}).join(' · ');}}
$('run').onchange=render;$('filter').oninput=()=>{kernelPage=0;operatorPage=0;renderRows();};$('reset').onclick=()=>setRange(0,R.duration_s);$('zin').onclick=()=>zoom(.5);$('zout').onclick=()=>zoom(2);$('left').onclick=()=>setRange(lo-(hi-lo)*.25,hi-(hi-lo)*.25);$('right').onclick=()=>setRange(lo+(hi-lo)*.25,hi+(hi-lo)*.25);$('apply').onclick=()=>setRange(Number($('start').value),Number($('end').value));
const findingMarkup=f=>'<article class="panel"><h3>'+esc(f.title)+'</h3><p>'+esc(f.text)+'</p><small>'+esc(f.kind)+' · '+esc((f.evidence_ids||[]).join(', '))+'</small></article>';
$('findings').innerHTML=D.findings.slice(0,3).map(findingMarkup).join('');$('more-findings').innerHTML=D.findings.slice(3).map(findingMarkup).join('');
table('sources',['ID','Tool / version','Artifact','Notes'],D.sources.map(s=>[esc(s.id),esc(s.tool)+' '+esc(s.version??''),s.artifact?'<a href="'+esc(s.artifact)+'" download rel="noopener noreferrer">'+esc(s.artifact)+'</a>':'—',esc(text(s.notes??''))]));render();
</script></body></html>'''


def build(data):
    data = derive(validate(data))
    payload = json.dumps(data, ensure_ascii=True, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return TEMPLATE.replace('__TITLE__', html.escape(data.get('title', 'H3 ref2va — Nsight benchmark'))).replace('__SCHEMA__', html.escape(__doc__)).replace('__DATA__', payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    try:
        result = build(json.loads(args.analysis.read_text(encoding='utf-8')))
        if args.analysis.resolve() == args.output.resolve(): raise ValueError('Output must differ from input')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding='utf-8')
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f'Invalid analysis or output: {exc}\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
