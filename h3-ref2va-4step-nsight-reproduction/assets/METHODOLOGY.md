# Ref2VA four-step reproduction: collection and accounting

This is a fresh reproduction of the reference report using its Larryvrh INT8 ConvRot Ref2VA checkpoint and v4-step600 LoRA. It does not use FastVideo FastH3 V2. The exact prompt and ordered trainer/trainee image hashes, seed 287987790, four-step simple schedule, native 736x1280 / 243-frame output, and final 720x1280 / 241-frame crop are retained. Both videos run at 24 FPS with native 32 kHz stereo audio.

## Execution and isolation

One complete warm-up precedes one measured generation in the same process. The GPU preflight is recorded in run.json: no compute PID, 4 MiB used, 0% utilization. The benchmark refuses to start above 64 MiB or with another compute process. Its 1-second GPU process monitor and independent target-100-ms NVML monitor detected only the benchmark process throughout the measured window. No unrelated GPU processes were terminated.

ComfyUI e377e263049f9338b4d12a3dd417b36ae62948ff and Turbo nodes 4274783a23afcfdbea3b4876cb79effd6c510785 are shared read-only. The Python environment is /home/raqeeb/uv/envs/h3_5090/bin/python (PyTorch 2.11.0+cu130). Source helpers expect to reside in the workspace nsight-work directory. Inputs are under nsight_reference/inputs. workflow.json contains the executed graph. Model checksums are collected after the timed run.

The runtime uses dynamic VRAM and two asynchronous offload streams, graph cache NONE, LoRA strength 1.0 and low_vram false. No explicit torch.compile. SageAttention 2.2.0 replaces diffusion attention only; text encoder and VAEs retain PyTorch attention. The original offload behavior is retained to reproduce the reference. The loader nodes run again in the measured pass; OS caches are not cleared. Warm-up, profiler postprocessing, preview conversion, HTML construction and upload are excluded from generation timings.

## Nsight Systems

```
--trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
--cuda-memory-usage=true --gpu-metrics-devices=0
--gpu-metrics-set=gb20x --gpu-metrics-frequency=10000
--capture-range=cudaProfilerApi --capture-range-end=stop
```

cudaProfilerStart begins immediately before the measured workflow; cudaProfilerStop follows crop/trim export. The displayed native-generation duration is CUDA-synchronized and includes Nsight overhead. Export is separately measured. ATen emit_nvtx annotations retain launch attribution.

Host NVTX spans are asynchronous, nested, inclusive wall spans and must not be summed. GPU-projected bounds use CUDA runtime correlation IDs and matching process/thread identity. ATen counts denote attributed GPU launches, not Python invocations. Kernel-presence union means elapsed time with at least one kernel, not occupancy or SM utilization. Summed kernel/copy durations may overlap and are not elapsed time.

GPU hardware observations describe intervals between timestamps. Exact phase and whole-run statistics use raw overlap-duration weighting; charts use 50-ms weighted means. Missing intervals are not filled with zero. Seven streams are displayed: SM active, tensor active, DRAM read/write, PCIe RX/TX, and GPC clock. GPC samples are in Hz despite the SQLite display label; the GB20x multiplier 1e-6 converts them to MHz. Hardware percentage activity differs from CUDA copy byte counts and bytes / summed copy duration. The expected warning about absent CPU scheduling data reflects --sample=none and --cpuctxsw=none. No scheduler-level CPU-state attribution is claimed.

## Host memory and device telemetry

The independent psutil/procfs/NVML child sampler targets 100 ms for RSS, CPU, device memory, power, temperature, I/O and host meminfo; PSS/private RAM targets 1 second. Raw timestamps and actual intervals are retained in memory_samples.csv. PSS/USS are deduplicated by their actual full-sample timestamp and clipped to the capture window. Sampled peaks can miss shorter excursions; memory categories must not be added or subtracted as though simultaneous.

System used equals MemTotal minus MemAvailable. It includes application, operating-system and profiler memory; file cache is partly reclaimable and shown separately. Process-tree RSS includes the sampler and export child and may count shared pages twice. PSS apportions shared pages; USS measures private resident pages. VmHWM is lifetime-wide and deliberately excluded. Global swap and I/O are not attributed exclusively to this process. Process CPU 100% means one logical CPU. Cumulative process-tree I/O may decrease when children exit.

PyTorch allocated/reserved peaks reset before the measured generation and exclude allocations outside its allocator. Device-wide NVML samples are the VRAM sizing evidence. No PyTorch allocator timeline is synthesized. Capacity scenarios compare measured whole-host peak with 64/96/128 GiB ceilings; the recommendation adds 32 GiB then rounds up to a 32-GiB increment. These are arithmetic estimates on a 186-GiB host, not tested memory limits or concurrency guarantees.

## Separate Nsight Compute diagnostic

Nsight Compute runs in a separate process with a DiT_step_3 NVTX filter, three representative kernel launches, detailed sections and kernel replay (21, 21 and 24 passes for the three kernels). Nsight warned that replay backed device memory up into system RAM and some launches were slow; all three selected samples completed. The full command is ncu_command.json. Its warm-up, replay duration and memory are excluded from the main benchmark. Raw .ncu-rep and CSV retain exact metric names and units. The report presents the same eight throughput/occupancy/cache metrics and top three not-issued warp-stall reasons as the reference. Stall fractions normalize sampled counts and are not percentages of total kernel elapsed time. Selected launches are not whole-model averages; a sampled SIMT SGEMM must not be generalized to the dominant INT8 GEMMs.

## Output and report verification

ffprobe verifies dimensions, frames, FPS and audio properties. Both native and exported MP4s pass full FFmpeg decoding. A WebM fallback and poster are encoded after profiling. Dialogue accuracy and lip sync are not formally scored, and matching inputs do not establish bit-identical output. Report CSS and renderer are retained from the reference; browser checks cover timeline controls, kernel filtering, video playback and viewport layout. All plotted results and counters come from this run rather than the reference's measurements.
