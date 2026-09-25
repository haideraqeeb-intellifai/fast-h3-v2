"""Pinned FastH3 V2 benchmark: one warm-up and one measured request."""
from pathlib import Path
import dataclasses
import gc
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'run'


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=str) + '\n')


class Monitor:
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        while not self.stop.is_set():
            try:
                self.samples.append({
                    'unix_time': time.time(),
                    'host_memory_kib': {line.split(':')[0]: int(line.split()[1])
                                        for line in Path('/proc/meminfo').read_text().splitlines()
                                        if line.split(':')[0] in ('MemAvailable', 'Shmem', 'SwapFree')},
                    'gpu': command('nvidia-smi', '--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu', '--format=csv,noheader,nounits'),
                    'processes': command('nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'),
                })
            except Exception as exc:
                self.samples.append({'error': str(exc)})
            self.stop.wait(1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / 'benchmark_executed.py').write_bytes(Path(__file__).read_bytes())
    before = command('nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version', '--format=csv')
    processes = command('nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader')
    if processes:
        raise RuntimeError(f'GPU already has compute processes: {processes}')
    save('gpu_before.json', {'gpu': before, 'compute_processes': processes})
    sys.path.insert(0, str(ROOT / 'work/FastVideo/examples/inference/basic'))
    import basic_fasth3
    import basic_fasth3_8step
    import torch
    from fastvideo import VideoGenerator
    prompt = (ROOT / 'reference/prompt.txt').read_text().strip()
    assert hashlib.sha256(prompt.encode()).hexdigest() == '3a652ed6b7fdf389ac5f328d3414c6a797d4c50bfb98232f3e6a41a3535f56e8'
    args = basic_fasth3_8step.parse_args([
        '--model-path', str(ROOT / 'models/FastH3-8-Step-V2'),
        '--prompt', prompt, '--output', str(OUT),
        '--height', '1280', '--width', '736', '--num-frames', '243',
        '--seed', '42', '--warmup-seed', '42', '--repeats', '1',
        '--num-gpus', '1', '--profile', 'all', '--vsa-kernel', 'triton',
        '--no-fa4', '--no-inference-torch-compile', '--no-torch-compile',
        '--no-compile-vae', '--no-parallel-vae', '--no-replicated-dit',
        '--no-lazy-module-load', '--h3-sequential-load', '--no-pin-cpu-memory',
    ])
    environment = basic_fasth3.configure_environment(args)
    environment.update({name: os.environ.get(name) for name in ('PYTORCH_ALLOC_CONF', 'OMP_NUM_THREADS')})
    basic_fasth3.validate_profile_dependencies(args)
    config = basic_fasth3.build_generator_config(args)
    config.engine.offload.dit_layerwise = True
    config.engine.offload.dit = False
    report = {
        'status': 'initializing', 'started_unix': time.time(),
        'task_started_utc': '2026-09-16T13:57:01Z',
        'model_id': 'FastVideo/FastVideo-FastH3-8-Step-V2',
        'model_revision': '3da2ddfe1954d9cda4c05b643dc0f26007a655c5',
        'fastvideo_commit': command('git', '-C', str(ROOT / 'work/FastVideo'), 'rev-parse', 'HEAD'),
        'runtime_patch_sha256': hashlib.sha256((ROOT / 'work/cpu-load.patch').read_bytes()).hexdigest(),
        'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'config': dataclasses.asdict(config), 'arguments': vars(args),
        'environment': environment, 'torch': torch.__version__,
        'runs': [],
    }
    save('report.json', report)
    generator = None
    try:
        started = time.perf_counter()
        generator = VideoGenerator.from_config(config)
        report['initialization_seconds'] = time.perf_counter() - started
        for phase in ('warmup', 'measured'):
            report['status'] = phase
            save('report.json', report)
            print(f'BENCHMARK {phase} START', flush=True)
            request = basic_fasth3.build_request(args, OUT / f'{phase}.mp4', 42)
            save(f'{phase}_request.json', dataclasses.asdict(request))
            with Monitor() as monitor:
                started = time.perf_counter()
                try:
                    result = generator.generate(request)
                    elapsed = time.perf_counter() - started
                finally:
                    save(f'{phase}_gpu_samples.json', monitor.samples)
            stages = getattr(result.logging_info, 'stages', {})
            entry = {
                'phase': phase, 'generation_seconds': elapsed,
                'pipeline_generation_seconds': result.generation_time,
                'stages': stages, 'video_path': result.video_path,
                'size': result.size, 'peak_memory_mb': result.peak_memory_mb,
                'sampled_device_peak_mib': max(float(s['gpu'].split(',')[0]) for s in monitor.samples if 'gpu' in s),
                'extra': result.extra,
            }
            report['runs'].append(entry)
            save('report.json', report)
            print(f'BENCHMARK {phase} COMPLETE {elapsed:.3f}s', flush=True)
            del result
            gc.collect()
        report['status'] = 'complete'
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        if generator is not None:
            generator.shutdown()
        report['finished_unix'] = time.time()
        save('report.json', report)


if __name__ == '__main__':
    main()
