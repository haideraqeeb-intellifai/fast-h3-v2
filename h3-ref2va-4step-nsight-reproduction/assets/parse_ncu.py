import csv,json,math
from pathlib import Path
root=Path(__file__).resolve().parent.parent
rows=list(csv.DictReader((root/'h3-ref2va-4step-nsight-reproduction/assets/ncu_raw.csv').open()))
units=next(r for r in rows if not r['ID']);rows=[r for r in rows if r['ID']]
assert len(rows)==3
metrics=[('sm__throughput.avg.pct_of_peak_sustained_elapsed','Compute / SM throughput'),('sm__warps_active.avg.pct_of_peak_sustained_active','Achieved occupancy'),('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed','Tensor pipe active cycles (elapsed denominator)'),('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed','Memory throughput'),('dram__cycles_active.avg.pct_of_peak_sustained_elapsed','DRAM active cycles'),('l1tex__throughput.avg.pct_of_peak_sustained_active','L1/TEX throughput (active denominator)'),('lts__throughput.avg.pct_of_peak_sustained_elapsed','L2 throughput'),('lts__t_sector_hit_rate.pct','L2 hit rate')]
def value(r,k):
 try:
  v=float(r[k].replace(',',''));return v if math.isfinite(v) else None
 except (ValueError,KeyError,TypeError):return None
out=[]
for r in rows:
 counters=[]
 for key,label in metrics:
  v=value(r,key);counters.append({'name':key,'label':label,'value':v,'unit':units.get(key,'%'),'status':'collected' if v is not None else 'unavailable','reason':'' if v is not None else 'Not returned in detailed set for this kernel/device'})
 stalls=[(k,value(r,k)) for k in r if k.startswith('smsp__pcsamp_warps_issue_stalled_') and k.endswith('_not_issued')];stalls=[(k,v) for k,v in stalls if v is not None];total=sum(v for k,v in stalls)
 for k,v in sorted(stalls,key=lambda x:-x[1])[:3]:
  counters.append({'name':k+' / sum(all not-issued reasons)','label':'Not-issued warp samples: '+k.removeprefix('smsp__pcsamp_warps_issue_stalled_').removesuffix('_not_issued'),'value':100*v/total if total else None,'unit':'% of not-issued samples','status':'derived' if total else 'unavailable','reason':'Normalized PC-sampling reason counts; not percentage of total kernel time'})
 name=r['Kernel Name'];note='Separate diagnostic execution, first matching launch in DiT step3; counter replay time and memory are excluded from benchmark aggregates. Nsight default clock/cache controls apply.'
 if 'simt_sgemm' in name:note+=' This matrix-multiply sample is FP32 SIMT. The dominant INT8 GEMMs are characterized by Systems timings; their per-kernel counters were not sampled.'
 out.append({'id':'ncu_'+r['ID'],'run_id':'measured','kernel_name':name,'source_id':'ncu','representative':True,'launch':{'diagnostic_id':r['ID'],'context':r['Context'],'stream':r['Stream'],'grid':r['Grid Size'],'block':r['Block Size'],'phase':'DiT_step_3'},'counters':counters,'notes':note})
(root/'nsight-work/ncu_summary.json').write_text(json.dumps(out,indent=2));print([(x['kernel_name'][:100],[(c['label'],c['value']) for c in x['counters'][:3]]) for x in out])
