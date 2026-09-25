from pathlib import Path
import json,math,statistics,re,csv
ROOT=Path(__file__).resolve().parent.parent
WORK=ROOT/'nsight-work'; OUT=ROOT/'h3-ref2va-4step-nsight-reproduction'; G=2**30
report=json.loads((OUT/'assets/run.json').read_text()); actual=report['runs'][0]
sysdata=json.loads((WORK/'systems.json').read_text()); rows=json.loads((OUT/'assets/memory_samples.json').read_text())
duration=max(sysdata['measured_duration_s'],(actual['capture_end_ns']-actual['capture_start_ns'])/1e9)
sources=[{'id':'nsys','tool':'Nsight Systems','version':'2026.4.1','artifact':'assets/measured.nsys-rep','notes':'CUDA/NVTX/OSRT; GB20x GPU counters at requested 10 kHz. Only measured capture window.'},{'id':'sqlite','tool':'Nsight Systems SQLite','artifact':'assets/measured.sqlite.gz'},{'id':'ram','tool':'psutil, Linux procfs and NVML','artifact':'assets/memory_samples.csv','notes':'Independent sampler. RSS/device telemetry target100ms; PSS/private target1s. Actual intervals retained. VmHWM excluded because it is lifetime-wide.'},{'id':'run','tool':'Measured benchmark and PyTorch','artifact':'assets/run.json'},{'id':'workflow','tool':'Exact executed workflow','artifact':'assets/workflow.json'},{'id':'verification','tool':'FFmpeg / ffprobe','artifact':'assets/verification.json'},{'id':'gpu_raw','tool':'Hardware samples','artifact':'assets/gpu_metrics_raw.csv.gz'},{'id':'phases','tool':'Exact phase statistics','artifact':'assets/phase_metrics.csv'},{'id':'ncu','tool':'Nsight Compute','version':'2026.2.1','artifact':'assets/kernels.ncu-rep','notes':'Separate diagnostic execution. Representative launches; kernel replay. Its elapsed time and RAM are excluded.'}]
sources.extend([{'id':'method','tool':'Collection and accounting notes','artifact':'assets/METHODOLOGY.md'}, {'id':'nsys_kernel_stats','tool':'NVIDIA built-in kernel statistics','artifact':'assets/nsys_stats_cuda_gpu_kern_sum.csv'}, {'id':'nsys_copy_time','tool':'NVIDIA built-in copy timing','artifact':'assets/nsys_stats_cuda_gpu_mem_time_sum.csv'}, {'id':'nsys_copy_size','tool':'NVIDIA built-in copy sizes','artifact':'assets/nsys_stats_cuda_gpu_mem_size_sum.csv'}, {'id':'nsys_api','tool':'NVIDIA CUDA API statistics','artifact':'assets/nsys_stats_cuda_api_sum.csv'}, {'id':'analysis','tool':'Report data','artifact':'assets/analysis.json'}, {'id':'ncu_csv','tool':'NVIDIA raw kernel counters','artifact':'assets/ncu_raw.csv'}])
model={
 'Base checkpoint':report['base_model']['file'],'LoRA':report['adapter']['file'],'LoRA strength / low_vram':'1.0 / false',
 'Text encoder':'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors','Video VAE':'minimax_h3_video_vae_fp16.safetensors','Audio VAE':'minimax_h3_audio_vae_fp32.safetensors',
 'Attention':'SageAttention2.2.0 diffusion only; PyTorch for text encoder and VAEs','Memory management':'Dynamic VRAM; two asynchronous offload streams; graph cache NONE; no explicit compilation',
 'Base revision':report['model_revision'],'LoRA revision':report['adapter']['revision'],'LoRA SHA-256':report['adapter']['sha256']}
workload={'Steps':4,'Sampler / scheduler':'MiniMaxH3TurboSampler / simple','Denoise / guidance':'1.0 / BasicGuider','Seed':287987790,'Native dimensions':'736 × 1280; 243 frames; 24 FPS; 10.125 seconds','Final export':'720 × 1280; 241 frames; 24 FPS; 10.041667 seconds','References':'trainer.png <Picture 1>, trainee.png <Picture 2>; ref_image_size=match','Audio':'Native generated; 32 kHz stereo','Prompt SHA-256':report['prompt_sha256'],'Scope':'One measured pass after a complete preparatory pass; export included. Single worker. OS caches not cleared.'}
host=json.loads((WORK/'host.json').read_text())
checks=json.loads((WORK/'model_checksums.json').read_text())
model['Base SHA-256']=next(c['sha256'] for c in checks if c['file']==report['base_model']['file'])
sources.extend([
 {'id':'checksums','tool':'Fresh SHA-256 model verification','artifact':'assets/model_checksums.json'},
 {'id':'inputs','tool':'Reference input hashes and provenance','artifact':'assets/input_provenance.json'},
 {'id':'comparison','tool':'Decoded output comparison with reference','artifact':'assets/reference_output_comparison.json'},
 {'id':'ncu_command','tool':'Exact Nsight Compute command','artifact':'assets/ncu_command.json'},
 {'id':'runner','tool':'Instrumented benchmark source','artifact':'assets/benchmark.py'}])
hardware={'GPU':'NVIDIA GeForce RTX5090','Driver':'595.84','VRAM GiB':32607/1024,'Host RAM GiB':rows[0]['system_total_bytes']/G,'CPU':host['cpu'],'Logical CPUs':host['logical_cpus'],'GPU isolation':'No competing compute PID detected during measured window'}
software={'PyTorch':report['packages']['torch'],'CUDA':report['torch_cuda'],'Python':report['python'],'ComfyUI commit':report['comfy_commit'],'Turbo nodes commit':report['turbo_commit'],'Nsight Systems':'2026.4.1','Nsight Compute':'2026.2.1','SageAttention':'2.2.0'}
phases=[]
for p in sysdata['phases']:
 if p['name']=='measured_run':continue
 x={'id':str(p['id']),'label':p['name'],'start_s':max(0,p['cpu_start_s']),'end_s':min(duration,p['cpu_end_s']),'basis':'Host NVTX interval; nested spans are inclusive','source_id':'nsys'}
 if p['gpu_projected_start_s'] is not None:
  x.update(gpu_start_s=max(0,p['gpu_projected_start_s']),gpu_end_s=min(duration,p['gpu_projected_end_s']),gpu_busy_s=sum(b-a for a,b in p['gpu_activity_union_intervals_s']))
 phases.append(x)
series=[]
def add_host(key,label,panel,unit='bytes',scope='process tree',full=False):
 samples=[];last=None
 for row in rows:
  t=row['time_s'];v=row.get(key)
  if full:
   stamp=row.get('full_sample_ns')
   if stamp==last:continue
   last=stamp;t=(stamp-actual['capture_start_ns'])/1e9
   if not 0<=t<=duration:continue
  if isinstance(v,(int,float)):samples.append([t,t,v])
 series.append({'id':key,'label':label,'panel':panel,'unit':unit,'scope':scope,'source_id':'ram','metric_name':key,'status':'collected','sampling':'instant','samples':samples})
for key,label in [('rss_bytes','Process-tree RSS'),('pss_bytes','Process-tree PSS'),('uss_bytes','Process-tree private RAM'),('system_used_bytes','Host used (total − available)'),('system_cached_bytes','Host file cache')]:add_host(key,label,'ram',scope='whole host' if key.startswith('system') else 'process tree',full=key in ['pss_bytes','uss_bytes'])
add_host('gpu_memory_bytes','Device VRAM','vram',scope='whole GPU')
add_host('cpu_percent','Process CPU (100% = one logical CPU)','cpu','%',scope='process tree')
add_host('system_cpu_percent','Host CPU utilization','cpu','%',scope='whole host')
add_host('gpu_power_w','GPU power','power','W',scope='whole GPU');add_host('gpu_temp_c','GPU temperature','temp','°C',scope='whole GPU')
add_host('swap_used_bytes','Host swap occupied','swap',scope='whole host')
# Cumulative read volume from application subtree; exited children can cause small discontinuities.
base=rows[0]['read_bytes']
series.append({'id':'disk_reads','label':'Cumulative physical reads since measured start','panel':'disk','unit':'bytes','scope':'process tree','source_id':'ram','metric_name':'read_bytes−initial','status':'collected','sampling':'instant','samples':[[r['time_s'],r['time_s'],max(0,r['read_bytes']-base)] for r in rows]})
selected={0:('gpc_clock','GPC clock','clock','MHz'),7:('sm_active','SMs Active','sm','%'),9:('tensor_active','Tensor Active','tensor','%'),19:('dram_read','DRAM read activity','dram','%'),20:('dram_write','DRAM write activity','dram','%'),21:('pcie_rx','PCIe RX (GPU receives)','pcie','%'),22:('pcie_tx','PCIe TX (GPU sends)','pcie','%')}
def convert_stats(st):
 return {'mean':st.get('mean'),'p95':st.get('p95'),'peak':st.get('max',st.get('peak')),'coverage_pct':100*st['coverage_fraction'] if st.get('coverage_fraction') is not None else None}
for s in sysdata['gpu_metrics']['series']:
 if s['metricId'] not in selected:continue
 sid,label,panel,unit=selected[s['metricId']]
 if s['metricId']==0:
  for z in s['intervals']:
   for j in range(2,min(5,len(z))):
    if z[j] is not None:z[j]/=1e6
  for st in [s['measured_statistics']]+[p[k] for p in s['phase_statistics'] for k in ['cpu_window','gpu_projected_window','gpu_activity_union']]:
   for key in ['mean','p95','max','peak']:
    if st.get(key) is not None:st[key]/=1e6
 points=s.get('intervals',s.get('chart_intervals',[]))
 raw_points=points
 points=[[max(0,z[0]),min(duration,z[1]),z[2]] for z in points if z[1]>z[0] and z[1]>0 and z[0]<duration]
 if 'max' not in s['measured_statistics'] and raw_points and len(raw_points[0])>=5:
  s['measured_statistics']['max']=max(z[4] for z in raw_points)
 ps={str(x['phase_id']):convert_stats(x['gpu_projected_window'] if next((p for p in phases if p['id']==str(x['phase_id']) and 'gpu_start_s' in p),None) else x['cpu_window']) for x in s['phase_statistics'] if any(p['id']==str(x['phase_id']) for p in phases)}
 series.append({'id':sid,'label':label,'panel':panel,'unit':unit,'scope':'whole GPU','source_id':'nsys','metric_name':s['name'],'status':'collected','sampling':'interval','samples':points,'statistics':convert_stats(s['measured_statistics']),'phase_statistics':ps,'raw_sample_count':s['measured_statistics'].get('interval_count')})
phase_summaries=[]
for p in phases:
 rr=[r for r in rows if p['start_s']<=r['time_s']<=p['end_s']]
 if rr:
  for key,label in [('rss_bytes','Process-tree RAM peak'),('system_used_bytes','Host used RAM peak'),('gpu_memory_bytes','Device VRAM peak')]:
   phase_summaries.append({'id':p['id']+'_'+key,'phase_id':p['id'],'label':p['label']+' · '+label,'value':max(r[key] for r in rr),'unit':'bytes','basis':'Peak observed sample inside host phase; no extrapolation','source_id':'ram'})
for key,label in [('peak_allocated_gib','PyTorch allocated peak'),('peak_reserved_gib','PyTorch reserved peak')]:phase_summaries.append({'id':key,'label':label,'value':actual[key]*G,'unit':'bytes','basis':'PyTorch counter reset before measured generation; excludes allocations outside PyTorch allocator','source_id':'run'})
# Kernel family classification shared with analyzer.
from analyze_sqlite import family
kernels=[{'id':'kernel_'+str(i),'name':k['name'],'family':family(k['name']),'calls':k['count'],'total_gpu_s':k['duration_sum_s'],'source_id':'nsys'} for i,k in enumerate(sysdata['kernels']['full_names'])]
operators=[{'id':'op_'+str(i),'name':k['name'],'calls':k['count'],'gpu_total_s':k['duration_sum_s'],'accounting':'GPU launches attributed to innermost ATen NVTX range; count is GPU launches, not Python calls','source_id':'nsys'} for i,k in enumerate(sysdata['aten_operators'])]
run={'id':'measured','kind':'measured','source_ids':['nsys','ram','run'],'profiler':'Nsight Systems + independent host sampler','duration_s':duration,'timing':[{'id':'generation','label':'Native generation','value_s':actual['generation_seconds'],'basis':'CUDA-synchronized generation wall time; profiler overhead included','source_id':'run'},{'id':'export','label':'Final export','value_s':actual['export_seconds'],'basis':'CPU crop/trim/export wall time','source_id':'run'},{'id':'total','label':'Generation + export','value_s':actual['generation_seconds']+actual['export_seconds'],'basis':'Measured end-to-end wall time','source_id':'run'},{'id':'gpu_busy','label':'Kernel presence (union)','value_s':sysdata['kernels']['totals']['presence_union_s'],'basis':'Time with ≥1 kernel; not SM utilization','source_id':'nsys'}],'phases':phases,'series':series,'kernels':kernels,'operators':operators,'phase_summaries':phase_summaries}
rss=max(r['rss_bytes'] for r in rows);pss=max(r['pss_bytes'] for r in rows);used=max(r['system_used_bytes'] for r in rows);vram=max(r['gpu_memory_bytes'] for r in rows)
recommended = math.ceil((used/G + 32)/32)*32
capacity = [
 {'id':'host_observed','label':'Observed host RAM use','resource':'RAM','run_id':'measured','observed':used,'ceiling':rows[0]['system_total_bytes'],'recommended':recommended*G,'unit':'bytes','basis':'Peak MemTotal minus MemAvailable inside measured window','evidence_ids':['ram'],'assumptions':'Single worker. Recommendation adds 32 GiB to the observed whole-host peak and rounds up to 32 GiB; this is not a tested memory limit.'},
 {'id':'app','label':'Application process-tree resident RAM','resource':'RAM','run_id':'measured','observed':rss,'ceiling':None,'unit':'bytes','basis':'Observed RSS peak; shared pages may be counted more than once','evidence_ids':['ram'],'interpretation':f'PSS peak {pss/G:.2f} GiB. Do not add process RSS to whole-host used RAM.'},
 {'id':'gpu','label':'GPU device memory','resource':'VRAM','run_id':'measured','observed':vram,'ceiling':32607*2**20,'unit':'bytes','basis':'NVML device-wide peak sample','evidence_ids':['ram'],'interpretation':'Includes allocations outside the PyTorch allocator; samples may miss shorter peaks.'}]
for amount in (64,96,128):
 capacity.append({'id':str(amount),'label':f'{amount} GiB capacity scenario','resource':'RAM','run_id':'measured','observed':used,'ceiling':amount*G,'unit':'bytes','basis':'Arithmetic comparison with this measured whole-host peak; untested capacity','evidence_ids':['ram'],'interpretation':f'{amount-used/G:.2f} GiB arithmetic headroom. Memory pressure and caching can change on a smaller host.'})
peak_row=max(rows,key=lambda r:r['rss_bytes'])
peak_phases=[p['label'] for p in phases if p['start_s']<=peak_row['time_s']<=p['end_s']]
busy=sysdata['kernels']['totals']['presence_union_s']
by_metric={s['id']:s for s in series}
attention=[k for k in kernels if k['family']=='attention']
attention_s=sum(k['total_gpu_s'] for k in attention)
kernel_s=sum(k['total_gpu_s'] for k in kernels)
findings=[
 {'title':'Reproduced Ref2VA workload','text':f"Native generation took {actual['generation_seconds']:.3f} s; export took {actual['export_seconds']:.3f} s. Exact reference prompt, trainer/trainee image hashes, seed 287987790, four steps, checkpoint, LoRA and runtime revisions were retained. This reproduces Larryvrh Ref2VA; it is not FastH3 V2.",'kind':'observation','evidence_ids':['run','workflow']},
 {'title':f'RAM sizing estimate: {recommended} GiB','text':f'Process-tree RSS peaked at {rss/G:.2f} GiB; PSS at {pss/G:.2f} GiB; host used RAM at {used/G:.2f} GiB. The recommendation adds 32 GiB of margin to the measured whole-host peak and rounds up. It is an estimate from this larger host, not a validated memory limit.','kind':'inference','evidence_ids':['ram']},
 {'title':'Measured GPU activity','text':f"Kernel presence covers {busy:.3f} s ({100*busy/duration:.2f}% of the measured window). Duration-weighted SM activity averages {by_metric['sm_active']['statistics']['mean']:.2f}%; tensor activity {by_metric['tensor_active']['statistics']['mean']:.2f}%. Kernel presence, SM activity and occupancy are different quantities.",'kind':'observation','evidence_ids':['nsys']},
 {'title':'Host-memory peak location','text':f"Peak RSS sample occurred at {peak_row['time_s']:.3f} s. Containing host phases: {', '.join(peak_phases) or 'outside named phases'}. Sampled peaks can miss briefer excursions.",'kind':'observation','evidence_ids':['ram','nsys']},
 {'title':'Attention kernel cost','text':f'Kernels classified as attention sum to {attention_s:.3f} s across {sum(k["calls"] for k in attention)} launches, or {100*attention_s/kernel_s:.2f}% of summed GPU kernel durations. Overlapping durations are not elapsed time or an achievable speedup estimate.','kind':'observation','evidence_ids':['nsys']},
 {'title':'Measurement scope','text':'One measured generation after a complete preparatory generation. Nsight overhead is included. CUDA/NVTX/OSRT capture and hardware counters cover the measured window; no CPU scheduling/backtrace samples were requested. Nested host phases are inclusive and must not be summed. Separate Nsight Compute replay is excluded from all main timing and memory statistics.','kind':'observation','evidence_ids':['run','nsys','ncu']},
 {'title':'Memory accounting','text':'RSS, PSS, private memory and whole-host used RAM are alternative views and must not be summed. File cache is partly reclaimable. The application tree includes telemetry and export subprocesses; external profiler memory is included in whole-host used RAM. Initial setup is outside the measured window.','kind':'observation','evidence_ids':['ram']},
 {'title':'Telemetry cadence','text':f"{len(rows)} lightweight samples; median interval {statistics.median(b['time_s']-a['time_s'] for a,b in zip(rows,rows[1:])):.3f} s; maximum gap {max(b['time_s']-a['time_s'] for a,b in zip(rows,rows[1:])):.3f} s. RSS/device telemetry targets 100 ms; PSS/private memory targets 1 s. Distinct full-sample timestamps determine the plotted full-memory observations.",'kind':'observation','evidence_ids':['ram']},
 {'title':'Runtime and media verification','text':'Four reference-conditioned transformer forwards and 200 completed Sage CUDA attention calls were verified. No recorded Sage fallbacks, errors or competing GPU compute PID. Native and final outputs passed full FFmpeg video/audio decoding. Final output is 720×1280, 241 frames, 24 FPS, 32 kHz stereo. Dialogue accuracy and lip sync were not formally scored.','kind':'observation','evidence_ids':['run','verification','ram']},
 {'title':'Allocator scope','text':'PyTorch allocated/reserved peaks were reset before the measured run. They exclude external allocations. Device memory is sampled through NVML; no allocator timeline was collected or synthesized.','kind':'observation','evidence_ids':['run','ram']},
 {'title':'RAM pressure and physical I/O','text':f"Peak sampled process swap: {max(r['process_swap_bytes'] for r in rows)/G:.3f} GiB. Application-tree physical reads increased by {(rows[-1]['read_bytes']-rows[0]['read_bytes'])/G:.2f} GiB; parent major faults increased by {rows[-1]['major_faults']-rows[0]['major_faults']:,}. Host-wide swap-in/out changes were {rows[-1]['swap_in_bytes']-rows[0]['swap_in_bytes']} / {rows[-1]['swap_out_bytes']-rows[0]['swap_out_bytes']} bytes; these are not attributed exclusively to the model.",'kind':'observation','evidence_ids':['ram']},
 {'title':'Clock and chart units','text':'GPC frequency is stored in Hz despite the SQLite MHz display label; the GB20x configuration multiplier 1e−6 converts it to MHz. Charts use 50 ms weighted means. Exact means, p95, peaks and coverage use raw intervals before binning. PCIe/DRAM hardware percentages are distinct from measured copy byte counts.','kind':'observation','evidence_ids':['gpu_raw','nsys']}]
for t in sysdata['cuda_transfers']['by_kind']:
 secs=t['duration_sum_s']; gb=t['bytes']/1e9
 findings.append({'title':f"CUDA transfer: {t['name']}",'text':f"{gb:.3f} GB across {t['count']:,} copies; summed GPU copy duration {secs:.3f} s; bytes divided by summed duration {gb/secs if secs else 0:.3f} GB/s. Copies may overlap compute or each other.",'kind':'observation','evidence_ids':['nsys']})
for warning in sysdata.get('warnings',[]):
 findings.append({'title':'Collection note','text':warning,'kind':'observation','evidence_ids':['nsys']})
findings.append({'title':'Reproduction agreement','text':'Fresh checkpoint and LoRA SHA-256 hashes match the reference. Decoded audio matches the reference exactly across all 314 frames. Video has 57 identical decoded frames out of 241 and FFmpeg SSIM 0.984699 against the reference export; it is not bit-identical. These checks describe reproduction agreement, not dialogue or identity quality.','kind':'observation','evidence_ids':['checksums','comparison']})
ncu=json.loads((WORK/'ncu_summary.json').read_text()) if (WORK/'ncu_summary.json').exists() else []
data={'schema_version':1,'title':'H3 ref2va · Four-step Nsight reproduction','model':model,'workload':workload,'hardware':hardware,'software':software,'sources':sources,'runs':[run],'ncu':ncu,'capacity':capacity,'findings':findings}
(WORK/'analysis.json').write_text(json.dumps(data,separators=(',',':'),allow_nan=False))
with (OUT/'assets/phase_metrics.csv').open('w') as f:
 from build_report import phase_window, statistics as series_statistics
 writer=csv.writer(f);writer.writerow(['phase_id','phase','metric','domain','mean','p95','peak','coverage_pct'])
 for p in phases:
  for s in series:
   start,end,domain=phase_window(s,p)
   st=series_statistics(s,start,end,p['id'])
   if st:writer.writerow([p['id'],p['label'],s['id'],domain,st.get('mean'),st.get('p95'),st.get('peak'),st.get('coverage_pct')])
print('Wrote analysis',len(series),'series',len(phases),'phases',len(kernels),'kernels')
