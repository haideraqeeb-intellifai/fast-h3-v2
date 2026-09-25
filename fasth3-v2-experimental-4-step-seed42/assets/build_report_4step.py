"""Verify the experimental four-step output and compare with the saved eight-step run."""
from pathlib import Path
import datetime
import hashlib
import html
import json
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'run_4step'
DEST = ROOT / 'fasth3-v2-experimental-4-step-seed42'
ASSETS = DEST / 'assets'


def cmd(*args):
    return subprocess.check_output(args, text=True)


def stage(run, match):
    return sum(v.get('execution_time', 0) for k, v in run['stages'].items()
               if match.lower() in (k + v.get('stage_class', '')).lower())


def main():
    started = time.perf_counter()
    ASSETS.mkdir(parents=True, exist_ok=True)
    report = json.loads((RUN / 'report.json').read_text())
    prior = json.loads((ROOT / 'run/report.json').read_text())
    assert report['status'] == prior['status'] == 'complete'
    assert report['prompt_sha256'] == prior['prompt_sha256']
    assert report['model_revision'] == prior['model_revision']
    for key, value in report['arguments'].items():
        if key not in ('model_path', 'output', 'steps'):
            assert value == prior['arguments'][key], (key, value, prior['arguments'][key])
    assert report['experiment']['dmd_denoising_steps'] == [999, 749, 500, 250]
    assert report['arguments']['steps'] == 5
    log = (RUN / 'run.log').read_text()
    assert '4 transformer forwards, DMD rungs=[999, 749, 500, 250]' in log
    verification = {}
    for entry in report['runs']:
        phase = entry['phase']
        path = Path(entry['video_path'])
        metadata = json.loads(cmd('ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)))
        video = next(s for s in metadata['streams'] if s['codec_type'] == 'video')
        audio = next(s for s in metadata['streams'] if s['codec_type'] == 'audio')
        assert (video['width'], video['height'], int(video['nb_frames']), video['r_frame_rate']) == (736, 1280, 243, '24/1')
        assert (int(audio['channels']), int(audio['sample_rate'])) == (2, 32000)
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'], check=True)
        hashes = {kind: cmd('ffmpeg', '-v', 'error', '-i', str(path), '-map', f'0:{kind}:0', '-f', 'hash', '-hash', 'sha256', '-').strip() for kind in ('v', 'a')}
        samples = json.loads((RUN / f'{phase}_gpu_samples.json').read_text())
        verification[phase] = {'metadata': metadata, 'full_decode': 'passed', 'decoded_hashes': hashes,
            'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'compute_pids': sorted({line.split(',')[0].strip() for s in samples for line in s.get('processes', '').splitlines() if line.strip()}),
            'minimum_host_available_gib': min(s['host_memory_kib']['MemAvailable'] for s in samples if 'host_memory_kib' in s) / 1024**2}
        shutil.copy2(path, ASSETS / f'{phase}.mp4')
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-ss', '1', '-i', str(ASSETS / 'measured.mp4'), '-frames:v', '1', '-q:v', '2', str(ASSETS / 'poster.jpg')], check=True)
    for source in RUN.glob('*.json'):
        shutil.copy2(source, ASSETS / source.name)
    for source in (RUN / 'benchmark_executed.py', RUN / 'run.log', ROOT / 'reference/prompt.txt', ROOT / 'work/cpu-load.patch', ROOT / 'prepare_four_step_model.py', Path(__file__)):
        shutil.copy2(source, ASSETS / source.name)
    assert hashlib.sha256((ASSETS / 'benchmark_executed.py').read_bytes()).hexdigest() == report['runner_sha256']
    shutil.copy2(ROOT / 'models/FastH3-V2-experimental-4-step/fastvideo_inference.json', ASSETS / 'experimental_schedule.json')
    shutil.copy2(ROOT / 'models/FastH3-8-Step-V2/fastvideo_inference.json', ASSETS / 'original_schedule.json')
    shutil.copy2(ROOT / 'models/FastH3-8-Step-V2/modular_model_index.json', ASSETS / 'modular_model_index.json')
    shutil.copy2(ROOT / 'run/report.json', ASSETS / 'eight_step_report.json')
    shutil.copy2(ROOT / 'run/measured.mp4', ASSETS / 'eight_step.mp4')
    (ASSETS / 'verification.json').write_text(json.dumps(verification, indent=2))
    warm, measured = report['runs']
    eight = prior['runs'][1]
    secs, prev = measured['generation_seconds'], eight['generation_seconds']
    rows = ''.join(f'<tr><th>{label}</th><td>{a:.3f} s</td><td>{b:.3f} s</td></tr>' for label, a, b in [
        ('Full generation', warm['generation_seconds'], secs),
        *[(label, stage(warm, match), stage(measured, match)) for label, match in [('Text encoding', 'Conditioning'), ('Denoising', 'Denoising'), ('Video decode', 'VideoDecoding'), ('Audio decode', 'AudioDecoding'), ('Frame processing', 'PostDecode'), ('MP4 saving', 'VideoSave'), ('Audio mux', 'AudioMux')]]])
    css = (ROOT / 'reference/index.html').read_text().split('<style>')[1].split('</style>')[0]
    prompt = html.escape((ROOT / 'reference/prompt.txt').read_text().strip())
    elapsed = time.time() - datetime.datetime.fromisoformat(report['task_started_utc'].replace('Z', '+00:00')).timestamp()
    process = report['finished_unix'] - report['started_unix']
    downloads = ''.join(f'<a href="assets/{p.name}" download>{p.name}</a>' for p in sorted(ASSETS.iterdir()) if p.suffix in ('.json', '.py', '.txt', '.log', '.patch'))
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>FastH3 V2 · Experimental four-step run</title><style>{css}</style></head><body><main>
<header><div class="eyebrow">FastH3 V2 · Matched-input experiment</div><h1>Four steps with the eight-step V2 checkpoint</h1><p class="intro">Same prompt, seed 42, geometry and runtime as the completed eight-step run. One warm-up followed by one measured generation.</p><span class="badge">Experimental schedule · Not trained for four steps</span></header>
<div class="stats"><div class="stat"><strong>{secs:.2f} s</strong><span>Measured full generation</span></div><div class="stat"><strong>{stage(measured,'Denoising'):.2f} s</strong><span>Measured denoising</span></div><div class="stat"><strong>{prev/secs:.2f}×</strong><span>Full-generation speedup over eight steps</span></div><div class="stat"><strong>4 forwards</strong><span>736 × 1280 · 10.125 s · 24 FPS</span></div></div>
<section><h2>Four-step output</h2><video controls playsinline preload="metadata" poster="assets/poster.jpg"><source src="assets/measured.mp4" type="video/mp4"></video><p><a href="assets/measured.mp4">Measured video</a> · <a href="assets/warmup.mp4">Warm-up video</a></p><p class="note">Native video with stereo audio. Complete decoding passed. Sampled warm-up frames show the hosts and taco sequence, but unwanted subtitles, microphone branding and lighter hair on the second host remain. This is a schedule experiment; visual fidelity, dialogue accuracy and lip sync have not been formally scored.</p></section>
<div class="grid"><section><h2>Four vs eight steps</h2><table><tr><th>Measurement</th><th>Four steps</th><th>Eight steps</th></tr><tr><th>Full generation</th><td>{secs:.3f} s</td><td>{prev:.3f} s</td></tr><tr><th>Denoising</th><td>{stage(measured,'Denoising'):.3f} s</td><td>{stage(eight,'Denoising'):.3f} s</td></tr><tr><th>Peak sampled GPU memory</th><td>{measured['sampled_device_peak_mib']:,.0f} MiB</td><td>{eight['sampled_device_peak_mib']:,.0f} MiB</td></tr></table><p>Full generation is {(1-secs/prev)*100:.1f}% faster in this pair of measurements. Denoising speedup: {stage(eight,'Denoising')/stage(measured,'Denoising'):.2f}×. One measurement per configuration; variability and equal quality are not established.</p><details><summary>View the earlier eight-step output</summary><video controls playsinline preload="none" src="assets/eight_step.mp4"></video></details></section>
<section><h2>Four-step timings</h2><table><tr><th>Component</th><th>Warm-up</th><th>Measured</th></tr>{rows}</table><p class="note">Full generation is wall time around the synchronous generate call, including request-time loading, text encoding, offloading, denoising, decoding and saving. Initialization ({report['initialization_seconds']:.3f} s) is separate. Text encoder reloads for each request; the first request also loads DiT/VAEs. Nested stage timings must not be summed.</p></section></div>
<section><h2>Exact schedule and configuration</h2><p>The published eight-forward ladder is <code>[999, 874, 749, 624, 500, 375, 250, 125]</code>. This experiment takes every other rung: <code>[999, 749, 500, 250]</code>, followed by terminal zero. Five sigma-grid points produce exactly four transformer forwards. Video/audio shifts remain 10/3.</p><p>The official eight-step entrypoint rejects this override. A separate local model view links the unchanged original weights and carries an explicitly marked experimental schedule. The original model contract remains intact. This is neither an official four-step recipe nor a separately distilled four-step checkpoint.</p><table><tr><th>Model</th><td>FastVideo/FastVideo-FastH3-8-Step-V2 · BF16</td></tr><tr><th>Inputs</th><td>Exact reference prompt · seed 42 · guidance 1.0 · empty negative prompt</td></tr><tr><th>Output</th><td>736 × 1280 · 243 frames · 24 FPS · 10.125 seconds · stereo 32 kHz</td></tr><tr><th>Hardware</th><td>One RTX 5090 · 32 GB</td></tr><tr><th>Attention</th><td>VSA-H3 · 80% sparsity · 64-token tiles · Triton</td></tr><tr><th>Runtime</th><td>Official H3 fused profile; no explicit DiT/VAE compilation; BF16 text encoder; sequential encoder release; DiT layerwise CPU offload; VAE CPU offload</td></tr></table><p class="note">The same CPU-first transformer loading patch from the eight-step run avoids initialization OOM. It changes weight placement only. Changing the schedule changes the numerical trajectory; seed parity does not imply equal quality.</p></section>
<section><h2>Does V2 support Ref2VA?</h2><p><strong>No supported distilled Ref2VA mode is provided by this checkpoint.</strong> The <a href="https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2#scope">official model card</a> states that it supports text-to-audio-video and that FL2VA and Ref2VA were not distilled. FastVideo has Ref2VA code, but this checkpoint's modular index points its <code>transformer_ref</code> component to <code>MiniMaxAI/MiniMax-H3</code>. Using that component would use the base model's reference transformer, not V2's distilled text-to-audio-video transformer. No Ref2VA generation was attempted.</p></section>
<section><h2>Verification and elapsed time</h2><p>Both outputs passed complete ffmpeg audio/video decoding and frame-count, geometry, frame-rate and audio checks. Warm-up/measured decoded video hashes match: <strong>{verification['warmup']['decoded_hashes']['v']==verification['measured']['decoded_hashes']['v']}</strong>; audio hashes match: <strong>{verification['warmup']['decoded_hashes']['a']==verification['measured']['decoded_hashes']['a']}</strong>. Device memory is sampled about once per second and may miss brief peaks. Per-sample process lists are included.</p><table><tr><th>Benchmark process, including initialization and both runs</th><td>{process:.2f} s</td></tr><tr><th>Follow-up inspection through report build</th><td>{elapsed:.2f} s ({elapsed/60:.2f} min)</td></tr><tr><th>Media verification and report assembly</th><td>BUILD_SECONDS s</td></tr></table><p class="note">Total follow-up time starts at 14:58:02 UTC, includes schedule investigation and preparation, and excludes earlier model installation/downloads and later checks. No publication time is included.</p></section>
<section><h2>Exact prompt</h2><details><summary>Read prompt</summary><pre>{prompt}</pre></details></section><section><h2>Evidence and reproducibility</h2><p class="provenance">Model revision: {report['model_revision']}<br>FastVideo commit: {report['fastvideo_commit']}<br>Prompt SHA-256: {report['prompt_sha256']}<br>Runner SHA-256: {report['runner_sha256']}</p><div class="downloads">{downloads}</div></section></main></body></html>'''
    build = time.perf_counter() - started
    (DEST / 'index.html').write_text(document.replace('BUILD_SECONDS', f'{build:.3f}'))
    timing = {'verification_and_build_seconds': build, 'task_to_report_seconds': elapsed, 'benchmark_process_seconds': process, 'measured_generation_seconds': secs, 'report_built_unix': time.time()}
    (ASSETS / 'report_timing.json').write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing, indent=2))
    print(DEST / 'index.html')


if __name__ == '__main__':
    main()
