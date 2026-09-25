(async () => {
 const assert = (condition, message) => { if (!condition) throw new Error(message); };
 const results = {title:document.title,series:D.runs[0].series.length,charts:document.querySelectorAll('#charts svg').length,steps:D.runs[0].phases.filter(p=>p.label.startsWith('DiT_step_')).length,ncu:D.ncu.length};
 assert(results.series===19,'Expected 19 series'); assert(results.charts===12,'Expected 12 charts'); assert(results.steps===4,'Expected four steps'); assert(results.ncu===3,'Expected three kernel counter sets');
 assert(D.runs[0].series.every(s=>s.samples.length>0),'Missing resource samples');
 const end=document.querySelector('#end');const original=end.value;
 document.querySelector('#zin').click();assert(Number(end.value)<Number(original),'Zoom failed');
 document.querySelector('#reset').click();assert(end.value===original,'Reset failed');
 const filter=document.querySelector('#filter');filter.value='qk_int_sv_f8_attn_kernel';filter.dispatchEvent(new Event('input'));
 assert(document.querySelector('#kernels').textContent.includes('qk_int_sv_f8_attn_kernel'),'Kernel filter failed');filter.value='';filter.dispatchEvent(new Event('input'));
 const v=document.querySelector('video');v.muted=true;await v.play();await new Promise(r=>setTimeout(r,900));
 assert(v.currentTime>0.3,'Video playback failed');assert(v.videoWidth===720&&v.videoHeight===1280,'Video geometry incorrect');
 results.video={time:v.currentTime,duration:v.duration,width:v.videoWidth,height:v.videoHeight,source:v.currentSrc};v.pause();
 results.noPageOverflow=document.documentElement.scrollWidth<=innerWidth;
 assert(results.noPageOverflow,'Page overflows viewport');results.controls=true;results.timing=D.runs[0].timing;
 return results;
})()
