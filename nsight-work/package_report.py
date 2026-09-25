from pathlib import Path
import json,re,shutil,hashlib
ROOT=Path(__file__).resolve().parent.parent
WORK=ROOT/'nsight-work';OUT=ROOT/'h3-ref2va-4step-nsight-reproduction';ASSETS=OUT/'assets'
data=json.loads((WORK/'analysis.json').read_text())
assert len(data['ncu'])==3
assert len(data['runs'][0]['series'])==19
assert len([p for p in data['runs'][0]['phases'] if p['label'].startswith('DiT_step_')])==4
for name in ['analysis.json','kernels.ncu-rep','ncu_command.json','nsys_command.json','ncu.log','model_checksums.json','analyze_sqlite.py','assemble_report.py','build_report.py','parse_ncu.py','prepare_artifacts.py','benchmark_ncu.py','telemetry.py','browser_check.js']:
 shutil.copy2(WORK/name,ASSETS/name)
for s in data['sources']:
 p=OUT/s['artifact'];assert p.is_file() and p.stat().st_size>0,(s['id'],str(p))
for name in ['measured.mp4','assets/preview.webm','assets/poster.jpg']:
 assert (OUT/name).stat().st_size>0
style=lambda s:re.search('<style>(.*?)</style>',s,re.S).group(1)
assert style((OUT/'index.html').read_text())==style((ROOT/'nsight_reference/index.html').read_text())
assert all(c['status']=='collected' for n in data['ncu'] for c in n['counters'][:8])
manifest=[]
for p in sorted(OUT.rglob('*')):
 if not p.is_file() or p.name=='artifact_manifest.json':continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 manifest.append({'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
(ASSETS/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'artifacts':len(manifest),'total_bytes':sum(x['bytes'] for x in manifest),'ncu_kernels':len(data['ncu']),'ncu_counters':[len(n['counters']) for n in data['ncu']],'matching_reference_css':True},indent=2))
