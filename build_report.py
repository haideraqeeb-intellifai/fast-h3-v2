"""Verify native outputs and build the shareable benchmark report."""
from pathlib import Path
import datetime
import hashlib
import html
import json
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'fasth3-8-step-v2-seed42'
ASSETS = DEST / 'assets'


def cmd(*args):
    return subprocess.check_output(args, text=True)


def main():
    started = time.perf_counter()
    ASSETS.mkdir(parents=True, exist_ok=True)
    report = json.loads((ROOT / 'run/report.json').read_text())
    assert report['status'] == 'complete'
    verification = {}
    for entry in report['runs']:
        phase = entry['phase']
        path = ROOT / 'run' / f'{phase}.mp4'
        metadata = json.loads(cmd('ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)))
        video = next(s for s in metadata['streams'] if s['codec_type'] == 'video')
        audio = next(s for s in metadata['streams'] if s['codec_type'] == 'audio')
        assert (video['width'], video['height'], int(video['nb_frames']), video['r_frame_rate']) == (736, 1280, 243, '24/1'), video
        assert int(audio['channels']) == 2, audio
        assert int(audio['sample_rate']) == 32000, audio
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'], check=True)
        hashes = {kind: cmd('ffmpeg', '-v', 'error', '-i', str(path), '-map', f'0:{kind}:0', '-f', 'hash', '-hash', 'sha256', '-').strip() for kind in ('v', 'a')}
        verification[phase] = {'metadata': metadata, 'full_decode': 'passed', 'decoded_hashes': hashes,
                               'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        samples = json.loads((ROOT / 'run' / f'{phase}_gpu_samples.json').read_text())
        verification[phase]['compute_pids'] = sorted({line.split(',')[0].strip()
            for sample in samples for line in sample.get('processes', '').splitlines() if line.strip()})
        verification[phase]['minimum_host_available_gib'] = min(
            sample['host_memory_kib']['MemAvailable'] for sample in samples if 'host_memory_kib' in sample) / 1024**2
        shutil.copy2(path, ASSETS / f'{phase}.mp4')
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-ss', '1', '-i', str(ASSETS / 'measured.mp4'), '-frames:v', '1', '-q:v', '2', str(ASSETS / 'poster.jpg')], check=True)
    for source in (ROOT / 'run').glob('*.json'):
        shutil.copy2(source, ASSETS / source.name)
    for source in (ROOT / 'benchmark.py', ROOT / 'build_report.py', ROOT / 'reference/prompt.txt', ROOT / 'run/run.log', ROOT / 'run/gpu_after.txt'):
        shutil.copy2(source, ASSETS / source.name)
    executed_runner = ROOT / 'run/benchmark_executed.py'
    if executed_runner.exists():
        shutil.copy2(executed_runner, ASSETS / 'benchmark.py')
    assert hashlib.sha256((ASSETS / 'benchmark.py').read_bytes()).hexdigest() == report['runner_sha256']
    for name in ('fastvideo_inference.json', 'checkpoint_content.json'):
        shutil.copy2(ROOT / 'models/FastH3-8-Step-V2' / name, ASSETS / name)
    for name in ('cpu-load.patch', 'checkpoint_verification.log', 'packages.txt', 'hardware.json'):
        shutil.copy2(ROOT / 'work' / name, ASSETS / name)
    (ASSETS / 'verification.json').write_text(json.dumps(verification, indent=2))
    warm, measured = report['runs']
    seconds = measured['generation_seconds']
    baseline = json.loads((ROOT / 'reference/report.json').read_text())['runs'][1]['generation_seconds']
    def stage(run, match):
        return sum(v.get('execution_time', 0) for k, v in run['stages'].items()
                   if match.lower() in (k + v.get('stage_class', '')).lower())
    timing_rows = [('Full generation', warm['generation_seconds'], seconds)]
    for label, match in [('Text encoding', 'Conditioning'), ('Denoising', 'Denoising'), ('Video decode', 'VideoDecoding'), ('Audio decode', 'AudioDecoding'), ('Frame processing', 'PostDecode'), ('MP4 saving', 'VideoSave'), ('Audio mux', 'AudioMux')]:
        timing_rows.append((label, stage(warm, match), stage(measured, match)))
    rows = ''.join(f'<tr><th>{label}</th><td>{a:.3f} s</td><td>{b:.3f} s</td></tr>' for label, a, b in timing_rows)
    config = [('Model', 'FastVideo/FastVideo-FastH3-8-Step-V2'), ('Sampling', '8 transformer forwards · 9 sigma points'), ('Seed / guidance', '42 / 1.0'), ('Resolution', '736 × 1280'), ('Frames / rate / duration', '243 / 24 FPS / 10.125 s'), ('Transformer / text encoder', 'BF16 / BF16 Qwen3-VL'), ('Attention', 'VSA-H3, 80% sparsity, 64-token tiles, Triton'), ('Scheduler shifts', 'Video 10 / audio 3'), ('Memory', 'DiT layerwise CPU offload; text encoder FSDP CPU offload and release after encoding; VAE CPU offload'), ('Compile / H3 fusions', 'Explicit DiT and VAE compile disabled / official H3 fusions enabled'), ('GPU', 'One RTX 5090, 32 GB')]
    config_rows = ''.join(f'<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>' for k,v in config)
    downloads = ''.join(f'<a href="assets/{p.name}" download>{p.name}</a>' for p in sorted(ASSETS.iterdir()) if p.suffix in ('.json', '.txt', '.py', '.log', '.patch'))
    prompt = html.escape((ROOT / 'reference/prompt.txt').read_text().strip())
    elapsed = time.time() - datetime.datetime.fromisoformat(report['task_started_utc']).timestamp()
    process_seconds = report['finished_unix'] - report['started_unix']
    reference_url = 'https://s3.renderplatform.com/agent-assets/Raqeeb/larryvrh-h3-v4-600-4steps/index.html'
    css = (ROOT / 'reference/index.html').read_text().split('<style>')[1].split('</style>')[0]
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>FastH3 8-Step V2 · matched-input benchmark</title><style>{css}</style></head><body><main>
<header><div class="eyebrow">MiniMax H3 · Reproducible run record</div><h1>FastH3 8-Step V2</h1><p class="intro">The same podcast-and-taco prompt, seed, resolution, and duration as the Larryvrh reference. One complete warm-up, then one measured generation on the same GPU model.</p><span class="badge">Completed · video and audio fully decoded</span></header>
<div class="stats"><div class="stat"><strong>{seconds:.2f} s</strong><span>Measured full generation</span></div><div class="stat"><strong>{stage(measured,'Denoising'):.2f} s</strong><span>Measured denoising</span></div><div class="stat"><strong>8 steps</strong><span>Seed 42 · guidance 1.0</span></div><div class="stat"><strong>736 × 1280</strong><span>10.125 s · 24 FPS · stereo audio</span></div></div>
<div class="grid"><div><section><h2>Measured output</h2><video controls playsinline preload="metadata" poster="assets/poster.jpg"><source src="assets/measured.mp4" type="video/mp4"></video><p class="video-caption"><a href="assets/measured.mp4">Download measured video</a> · <a href="assets/warmup.mp4">Warm-up video</a></p><p class="note">Native, untrimmed output. Sampled frames show unwanted subtitles and a microphone emblem despite the prompt; the second host also has lighter hair than specified. Dialogue accuracy, lip sync, and exact shot timing have not been formally scored.</p></section><section><h2>Configuration</h2><table>{config_rows}</table></section></div>
<div><section><h2>Benchmark timings</h2><table class="timings"><thead><tr><th>Component</th><th>Warm-up</th><th>Measured</th></tr></thead><tbody>{rows}</tbody></table><p class="note">Full generation is host wall time around the synchronous generate call, including request-time model loading/reloading, encoding, offloading, denoising, decoding, saving and audio mux. The text encoder is reloaded and released for each subsequent request; the first request also loads DiT/VAEs after encoding. Pipeline stages use CUDA-synchronized wall time; nested times must not be summed. Initialization ({report['initialization_seconds']:.3f} s) and setup/downloads are excluded.</p><p><strong>{243/seconds:.3f}</strong> output frames per second · <strong>{seconds/10.125:.2f}×</strong> output duration in generation time.</p></section>
<section><h2>Comparison with the reference</h2><table><tr><th>Larryvrh v4-600 · 4 steps</th><td>{baseline:.3f} s</td></tr><tr><th>FastH3 V2 · 8 steps</th><td>{seconds:.3f} s</td></tr><tr><th>V2 / reference latency</th><td>{seconds/baseline:.2f}×</td></tr></table><p>The prompt, seed, output geometry, warm-up procedure, and GPU model match. This compares two complete inference configurations: the reference used an INT8 ConvRot model, NVFP4-AWQ text encoder, dense attention, ComfyUI, and four steps. V2 uses the published BF16 checkpoint, BF16 encoder, VSA-H3, FastVideo, and eight steps. It is not an isolated checkpoint-speed comparison or an equal-quality claim.</p><p class="note">A local initialization patch stages transformer weights on CPU before attaching the existing layerwise offload hooks. The unmodified loader exhausted 32 GB VRAM before generation. The patch changes weight placement only and is included below. An additional attempt saved a warm-up but lost its worker during measured denoising; host-memory pressure was suspected, not confirmed by kernel logs. The final run uses sequential text-encoder release and disables optional CPU memory pinning (DiT layerwise hooks still pin their weights). The unfused sequential attempt then exceeded GPU memory in SwiGLU, so the final configuration uses the official H3 fused inference profile and expandable CUDA allocations. Fusions can change floating-point operation order. Earlier attempts and waiting for an unrelated GPU job are excluded from final benchmark timings but included in total task time.</p><a href="{reference_url}">Open the reference report</a></section>
<section><h2>GPU and verification</h2><p>Sampled peak device memory: warm-up {warm['sampled_device_peak_mib']:,.0f} MiB; measured {measured['sampled_device_peak_mib']:,.0f} MiB. Sampling interval is about one second and can miss brief peaks. Per-sample process lists are included in the evidence.</p><p>Minimum available host RAM: warm-up {verification['warmup']['minimum_host_available_gib']:.2f} GiB; measured {verification['measured']['minimum_host_available_gib']:.2f} GiB. Observed compute PIDs: warm-up {html.escape(str(verification['warmup']['compute_pids']))}; measured {html.escape(str(verification['measured']['compute_pids']))}.</p><p>Both outputs passed complete audio/video decoding and contain 243 frames at 24 FPS, 736 × 1280, and stereo audio. Decoded video hashes match: <strong>{verification['warmup']['decoded_hashes']['v'] == verification['measured']['decoded_hashes']['v']}</strong>; audio hashes match: <strong>{verification['warmup']['decoded_hashes']['a'] == verification['measured']['decoded_hashes']['a']}</strong>.</p><p class="note">One measured sample; timing variance is unknown. CUDA deterministic algorithms were not enforced. Matching seed values across runtimes do not guarantee identical initial noise or output.</p></section></div></div>
<section><h2>Time to produce this report</h2><table><tr><th>Benchmark process, including initialization and both runs</th><td>{process_seconds:.2f} s</td></tr><tr><th>Elapsed from initial inspection to report built</th><td>{elapsed:.2f} s ({elapsed/60:.2f} min)</td></tr><tr><th>Verification and report build</th><td>BUILD_SECONDS s</td></tr></table><p class="note">The elapsed task time includes investigation, environment installation, checkpoint download, retries, GPU availability waiting, the benchmark, and checks completed before this report build. Publication and later checks are excluded. The build-only measurement covers media verification and HTML assembly; generation timing excludes all report preparation.</p></section>
<section><h2>Exact generation prompt</h2><details><summary>Read the full prompt</summary><pre>{prompt}</pre></details></section>
<section><h2>Evidence and reproducibility</h2><p class="provenance">Model revision: {report['model_revision']}<br>FastVideo commit: {report['fastvideo_commit']}<br>Prompt SHA-256: {report['prompt_sha256']}<br>Runner SHA-256: {report['runner_sha256']}</p><div class="downloads">{downloads}</div></section>
<footer>Generated from saved measurements · <a href="https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2">Model card</a></footer></main></body></html>'''
    build_seconds = time.perf_counter() - started
    (DEST / 'index.html').write_text(document.replace('BUILD_SECONDS', f'{build_seconds:.3f}'))
    timing = {'verification_and_build_seconds': build_seconds, 'task_to_report_seconds': elapsed,
              'benchmark_process_seconds': process_seconds, 'measured_generation_seconds': seconds,
              'report_built_unix': time.time()}
    (ASSETS / 'report_timing.json').write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing, indent=2))
    print(DEST / 'index.html')


if __name__ == '__main__':
    main()
