import sys,json,time,os
from pathlib import Path
import psutil,pynvml
pid=int(sys.argv[1]); dest=Path(sys.argv[2]); stop=Path(sys.argv[3]); target=psutil.Process(pid)
pynvml.nvmlInit(); gpu=pynvml.nvmlDeviceGetHandleByIndex(0)
f=dest.open('w',buffering=1); last_full=0; full={}; procs={}; errors=[]
def attempt(fn):
 try:return fn()
 except Exception:return None
while not stop.exists():
 t=time.perf_counter(); now=time.perf_counter_ns()
 try:
  children=target.children(recursive=True); allp=[target]+children
  rss=vms=read=write=cpu=0; live=[]
  for p in allp:
   try:
    mi=p.memory_info(); io=p.io_counters(); rss+=mi.rss;vms+=mi.vms;read+=io.read_bytes;write+=io.write_bytes
    if p.pid not in procs:procs[p.pid]=p
    cpu+=procs[p.pid].cpu_percent();live.append(p.pid)
   except psutil.Error:pass
  if t-last_full>=1:
   pss=uss=swap=0
   for p in allp:
    try:
     m=p.memory_full_info();pss+=m.pss;uss+=m.uss;swap+=m.swap
    except psutil.Error:pass
   full={'pss_bytes':pss,'uss_bytes':uss,'process_swap_bytes':swap,'full_sample_ns':now};last_full=t
  vm=psutil.virtual_memory();sw=psutil.swap_memory(); mem={}
  for line in Path('/proc/meminfo').read_text().splitlines():
   key,val=line.split(':',1);mem[key]=int(val.strip().split()[0])*1024
  stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  status=Path(f'/proc/{pid}/status').read_text(); hwm=next((int(l.split()[1])*1024 for l in status.splitlines() if l.startswith('VmHWM:')),None)
  m=pynvml.nvmlDeviceGetMemoryInfo(gpu);u=pynvml.nvmlDeviceGetUtilizationRates(gpu)
  compute=attempt(lambda:[{'pid':x.pid,'bytes':x.usedGpuMemory} for x in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu)])
  row={'monotonic_ns':now,'rss_bytes':rss,'vms_bytes':vms,**full,'vmhwm_bytes':hwm,'system_total_bytes':vm.total,'system_available_bytes':vm.available,'system_used_bytes':vm.total-vm.available,'system_cached_bytes':mem.get('Cached',0),'system_anon_bytes':mem.get('AnonPages',0),'system_sreclaimable_bytes':mem.get('SReclaimable',0),'system_mlocked_bytes':mem.get('Mlocked',0),'swap_used_bytes':sw.used,'swap_in_bytes':sw.sin,'swap_out_bytes':sw.sout,'read_bytes':read,'write_bytes':write,'minor_faults':int(stat[7]),'major_faults':int(stat[9]),'cpu_percent':cpu,'gpu_memory_bytes':m.used,'gpu_util_percent':u.gpu,'gpu_memory_util_percent':u.memory,'gpu_power_w':attempt(lambda:pynvml.nvmlDeviceGetPowerUsage(gpu)/1000),'gpu_temp_c':attempt(lambda:pynvml.nvmlDeviceGetTemperature(gpu,0)),'gpu_clock_mhz':attempt(lambda:pynvml.nvmlDeviceGetClockInfo(gpu,0)),'compute_processes':compute,'pids':live,'system_cpu_percent':psutil.cpu_percent()}
  f.write(json.dumps(row)+'\n')
 except Exception as e:f.write(json.dumps({'monotonic_ns':now,'error':repr(e)})+'\n')
 time.sleep(max(0,0.1-(time.perf_counter()-t)))
f.close()
