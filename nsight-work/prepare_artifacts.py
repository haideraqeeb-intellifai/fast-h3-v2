from pathlib import Path
import csv
import gzip
import hashlib
import json
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / 'nsight-work'
RUN = ROOT / 'nsight-run'
OUT = ROOT / 'h3-ref2va-4step-nsight-reproduction'
ASSETS = OUT / 'assets'
ASSETS.mkdir(parents=True, exist_ok=True)
report = json.loads((RUN / 'report.json').read_text())
assert report['status'] == 'complete'
assert len(report['runs']) == 1
actual = report['runs'][0]
assert report['seed'] == 287987790 and report['steps'] == 4
assert actual['forward_counts']['reference_conditioned_forwards'] == 4
assert actual['forward_counts']['sage_qk_int8_pv_fp8_cuda_completed'] == 200
assert not actual['forward_counts'].get('sage_pytorch_fallbacks')
assert not actual['forward_counts'].get('sage_errors')
rows = []
for line in (RUN / 'telemetry.jsonl').read_text().splitlines():
    row = json.loads(line)
    if actual['capture_start_ns'] <= row['monotonic_ns'] <= actual['capture_end_ns']:
        assert 'error' not in row, row
        row['time_s'] = (row['monotonic_ns'] - actual['capture_start_ns']) / 1e9
        rows.append(row)
assert rows
own_pids = {int(x) for r in json.loads((RUN / 'measured_1_gpu_samples.json').read_text())
            for x in r.get('compute_pids', '').splitlines() if x.strip()}
assert len(own_pids) == 1, own_pids
for row in rows:
    assert {p['pid'] for p in row['compute_processes']} <= own_pids, row['compute_processes']
(ASSETS / 'memory_samples.json').write_text(json.dumps(rows))
keys = sorted(set().union(*(r.keys() for r in rows)))
with (ASSETS / 'memory_samples.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in rows)
for source, name in [(RUN/'report.json', 'run.json'), (RUN/'measured_1_workflow.json','workflow.json'),
                     (RUN/'benchmark.py','benchmark.py'),(RUN/'measured_1_gpu_samples.json','gpu_samples.json'),
                     (ROOT/'nsight_reference/prepared_inputs.json','input_provenance.json')]:
    shutil.copy2(source, ASSETS/name)
for p in (ROOT/'nsight_reference/inputs').iterdir():
    shutil.copy2(p, ASSETS/p.name)
shutil.copy2(RUN/'measured.mp4', OUT/'measured.mp4')
shutil.copy2(Path(actual['video']), ASSETS/'measured_native.mp4')
verification = {}
for name, expected in [('measured.mp4', (720,1280,241)), ('assets/measured_native.mp4',(736,1280,243))]:
    path = OUT/name
    metadata = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)],text=True))
    video = next(s for s in metadata['streams'] if s['codec_type']=='video')
    audio = next(s for s in metadata['streams'] if s['codec_type']=='audio')
    assert (video['width'],video['height'],int(video['nb_frames'])) == expected
    assert video['r_frame_rate']=='24/1' and int(audio['sample_rate'])==32000 and audio['channels']==2
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-f','null','-'],check=True)
    verification[name] = {'metadata':metadata, 'full_decode':'passed','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(ASSETS/'verification.json').write_text(json.dumps(verification,indent=2))
cpu = next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
import os
(WORK/'host.json').write_text(json.dumps({'cpu':cpu,'logical_cpus':os.cpu_count(),'platform':platform.platform()},indent=2))
for name in ('measured.nsys-rep','gpu_metrics_raw.csv.gz'):
    if (WORK/name).exists(): shutil.copy2(WORK/name,ASSETS/name)
if (WORK/'measured.sqlite').exists():
    with (WORK/'measured.sqlite').open('rb') as src, gzip.open(ASSETS/'measured.sqlite.gz','wb',compresslevel=1) as dst:
        shutil.copyfileobj(src,dst)
print(json.dumps({'samples':len(rows),'native_generation_seconds':actual['generation_seconds'],'export_seconds':actual['export_seconds'],'compute_pids':sorted(own_pids)},indent=2))
