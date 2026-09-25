"""Isolated transfer measurements, not a rerun of the video-generation benchmark."""
from pathlib import Path
from types import SimpleNamespace
import gc
import json
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'decoder_transfer_measurements.json'


def main():
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
    if processes:
        raise RuntimeError(f'GPU busy: {processes}')
    import torch
    from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
    from fastvideo.models.loader.component_loader import VAELoader, AudioDecoderLoader
    torch.set_num_threads(8)
    args = SimpleNamespace(vae_cpu_offload=True, model_paths={}, pipeline_config=MiniMaxH3PipelineConfig())
    result = {'method': 'Actual checkpoint-loaded FP32 VAE modules; synchronous module.to(device), CUDA synchronized before/after; one excluded round trip then five measurements. No decoding or denoising. Not an in-pipeline attribution.',
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(), 'components': {}}
    for component, loader in [('vae', VAELoader()), ('audio_vae', AudioDecoderLoader())]:
        print(f'Loading {component}', flush=True)
        model = loader.load(str(ROOT / 'models/FastH3-8-Step-V2' / component), args)
        tensors = list(model.parameters()) + list(model.buffers())
        size = sum(t.numel() * t.element_size() for t in tensors)
        del tensors
        samples = []
        with torch.inference_mode():
            for repeat in range(6):
                timings = {}
                for device in ('cuda', 'cpu'):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    model.to(device)
                    torch.cuda.synchronize()
                    timings['cpu_to_gpu_s' if device == 'cuda' else 'gpu_to_cpu_s'] = time.perf_counter() - started
                if repeat:
                    samples.append(timings)
                print(component, repeat, timings, flush=True)
        result['components'][component] = {'bytes': size, 'gib': size / 1024**3, 'samples': samples,
            'median': {key: statistics.median(s[key] for s in samples) for key in samples[0]},
            'min': {key: min(s[key] for s in samples) for key in samples[0]},
            'max': {key: max(s[key] for s in samples) for key in samples[0]}}
        OUT.write_text(json.dumps(result, indent=2) + '\n')
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
