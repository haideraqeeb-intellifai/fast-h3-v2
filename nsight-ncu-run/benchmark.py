"""Larryvrh H3 Turbo: one warm-up followed by one measured generation."""

import argparse
import asyncio
from collections import Counter
from fractions import Fraction
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import threading
import uuid
import contextlib

ROOT = Path(__file__).resolve().parent.parent
COMMIT = "e377e263049f9338b4d12a3dd417b36ae62948ff"
TURBO_COMMIT = "4274783a23afcfdbea3b4876cb79effd6c510785"
LORA_REVISION = "43a74557ac3f6539db8e0f2a959d03feb7a81480"
MODEL_REVISION = "a98869194787969724c7425d95d0ed73ce9202af"
MODELS = {
    "diffusion_models": ["minimax_h3_fl2va_int8_convrot.safetensors"],
    "text_encoders": ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],
    "loras": ["minimax_h3_turbo_v4_step600_ema.safetensors"],
    "vae": ["minimax_h3_video_vae_fp16.safetensors", "minimax_h3_audio_vae_fp32.safetensors"],
}


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def graph(prompt, seed, phase, steps=6, attention_backend="pytorch"):
    workflow = {}

    def node(name, kind, **inputs):
        workflow[name] = {"class_type": kind, "inputs": inputs}
        return [name, 0]

    model = node("diffusion", "UNETLoader", unet_name=MODELS["diffusion_models"][0], weight_dtype="default")
    clip = node("text", "CLIPLoader", clip_name=MODELS["text_encoders"][0], type="minimax", device="default")
    vae = node("video_vae", "VAELoader", vae_name=MODELS["vae"][0])
    audio_vae = node("audio_vae", "VAELoader", vae_name=MODELS["vae"][1])
    conditioning = node("conditioning", "BenchmarkReference", clip=clip, vae=vae, audio_vae=audio_vae,
                        prompt=prompt, width=736, height=1280, length=243)
    model = node("adapter", "MiniMaxH3TurboLoRA", model=model, lora_name=MODELS["loras"][0], strength=1.0, low_vram=False)
    if attention_backend == "sage":
        model = node("attention", "BenchmarkSage", model=model)
    noise = node("noise", "RandomNoise", noise_seed=seed)
    guider = node("guider", "BasicGuider", model=model, conditioning=conditioning)
    sampler = node("sampler", "MiniMaxH3TurboSampler")
    sigmas = node("schedule", "BasicScheduler", model=model, scheduler="simple", steps=steps, denoise=1.0)
    samples = node("denoise", "SamplerCustomAdvanced", noise=noise, guider=guider,
                   sampler=sampler, sigmas=sigmas, latent_image=["conditioning", 1])
    images = node("video_decode", "VAEDecode", samples=samples, vae=vae)
    audio = node("audio_decode", "VAEDecodeAudio", samples=samples, vae=audio_vae)
    video = node("video", "CreateVideo", images=images, audio=audio, fps=24.0, bit_depth=8)
    node("save", "SaveVideo", video=video, filename_prefix=phase + "/trainer_trainee", format="mp4", codec="auto")
    return workflow


class Measurements:
    def __init__(self, torch):
        self.torch = torch
        self.calls = []
        self.originals = []

    def wrap(self, owner, method, label):
        original = inspect.getattr_static(owner, method)
        function = original.__func__ if isinstance(original, (classmethod, staticmethod)) else original

        @functools.wraps(function)
        def measured(*args, **kwargs):
            start_ns = time.perf_counter_ns()
            self.torch.cuda.nvtx.range_push(label)
            start = time.perf_counter()
            try:
                result = function(*args, **kwargs)
                return result
            finally:
                self.torch.cuda.nvtx.range_pop()
                elapsed = time.perf_counter() - start
                self.calls.append({"component": label, "seconds": elapsed, "start_ns": start_ns, "end_ns": time.perf_counter_ns()})
                print(f"{label}: {elapsed:.3f} seconds", flush=True)

        replacement = type(original)(measured) if isinstance(original, (classmethod, staticmethod)) else measured
        setattr(owner, method, replacement)
        self.originals.append((owner, method, original))

    def close(self):
        for owner, method, original in reversed(self.originals):
            setattr(owner, method, original)


class Progress:
    client_id = None
    last_node_id = None

    def send_sync(self, event, data, sid=None):
        if event == "execution_error":
            print(json.dumps(data), flush=True)


def check_checkout(path, expected):
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output(["git", "-C", str(path), "diff", "HEAD", "--stat"], text=True).strip()
    if actual != expected or changes:
        raise RuntimeError(f"Expected unmodified {expected} at {path}; found {actual}, changes={changes}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=ROOT.parent / "baseline/ComfyUI")
    parser.add_argument("--model-root", type=Path, default=ROOT / "models")
    parser.add_argument("--turbo-root", type=Path, default=ROOT / "upstream")
    parser.add_argument("--prompt", type=Path, default=ROOT / "prompt.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention", choices=["pytorch", "sage"], default="pytorch")
    parser.add_argument("--steps", type=int, choices=[4, 6, 8], default=6)
    parser.add_argument("--base-name", choices=["minimax_h3_fl2va_int8_convrot.safetensors", "minimax_h3_ref2va_pruned_int8_convrot.safetensors"], default=MODELS["diffusion_models"][0])
    parser.add_argument("--lora-name", choices=["minimax_h3_turbo_v4_step600_ema.safetensors", "minimax_h3_turbo_4step_ema_ckpt850.safetensors"], default=MODELS["loras"][0])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    MODELS["loras"] = [args.lora_name]
    MODELS["diffusion_models"] = [args.base_name]
    if args.runs < 1 or not 0 <= args.seed < 2**64:
        parser.error("runs must be positive and seed must be an unsigned 64-bit integer")
    out = (args.output or ROOT / "runs" / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])).resolve()
    out.mkdir(parents=True, exist_ok=False)
    prompt = args.prompt.read_text().strip()
    workflow = graph(prompt, args.seed, "warmup", args.steps, args.attention)
    save(out / "workflow.json", workflow)
    (out / "benchmark.py").write_bytes(Path(__file__).read_bytes())
    (out / "prompt.txt").write_text(prompt + "\n")
    report = {"status": "preflight", "runs": [], "seed": args.seed, "width": 736,
              "height": 1280, "frames": 243, "fps": 24, "duration_seconds": 10.125,
              "base_model": {"file": args.base_name, "pruned": "pruned" in args.base_name},
              "comfy_commit": COMMIT, "model_revision": MODEL_REVISION,
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "timing_method": "Measured run only after one warm-up. NVTX host spans are not CUDA-synchronized; GPU projection derived from Systems. Whole generation ends with CUDA synchronization. OS caches not cleared. Profiling overhead included.",
              "graph_cache": "NONE", "explicit_compile": False, "upload_seconds": 0}
    save(out / "report.json", report)
    print(f"OUTPUT={out}", flush=True)
    if args.dry_run:
        report["status"] = "dry_run"
        save(out / "report.json", report)
        return
    meter = None
    try:
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"], text=True).strip()
        report["gpu_preflight_processes"] = processes
        report["gpu_preflight"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv"], text=True)
        if processes:
            raise RuntimeError(f"GPU is occupied; no run started: {processes}")
        used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
        if used > 64:
            raise RuntimeError(f"GPU is not empty: {used} MiB allocated")
        check_checkout(args.comfy_root, COMMIT)
        check_checkout(args.turbo_root, TURBO_COMMIT)
        files = [args.model_root / folder / name for folder, names in MODELS.items() for name in names]
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(path)
        report["models"] = [{"path": str(p.resolve()), "bytes": p.stat().st_size,
                             "verification": "Existing local file; content hash not verified"} for p in files]
        report["packages"] = dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata["Name"]))
        report["hardware"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"], text=True)
        report["python"] = sys.version
        report["platform"] = platform.platform()
        sys.path.insert(0, str(args.comfy_root.resolve()))
        sys.argv = [sys.argv[0], "--use-pytorch-cross-attention", "--enable-dynamic-vram"]
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run with GPU access")
        report["torch_cuda"] = torch.version.cuda
        start = time.perf_counter()
        from comfy.cli_args import args as comfy_args
        comfy_args.models_directory = str(args.model_root.resolve())
        comfy_args.disable_all_custom_nodes = True
        comfy_args.cache_none = True
        for name in ("input", "output", "temp", "user"):
            directory = out / "runtime" / name
            directory.mkdir(parents=True)
            setattr(comfy_args, name + "_directory", str(directory))
        import main as comfy_main
        import nodes
        import execution
        import comfy.sd
        import comfy.samplers
        import comfy.ldm.modules.attention as attention
        counts = Counter()
        async def initialize():
            await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
            if not await nodes.load_custom_node(str(args.turbo_root.resolve()), module_parent="benchmark_turbo"):
                raise RuntimeError("Turbo node registration failed")
        asyncio.run(initialize())
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        from PIL import Image
        import numpy as np
        class BenchmarkReference:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"clip": ("CLIP",), "vae": ("VAE",), "audio_vae": ("VAE",),
                    "prompt": ("STRING", {"multiline": True}), "width": ("INT",), "height": ("INT",), "length": ("INT",)}}
            RETURN_TYPES = ("CONDITIONING", "LATENT")
            FUNCTION = "execute"
            CATEGORY = "benchmark"
            def execute(self, clip, vae, audio_vae, prompt, width, height, length):
                tensors = {}
                for index, name in enumerate(["trainer.png", "trainee.png"], 1):
                    image = Image.open(ROOT / "nsight_reference/inputs" / name).convert("RGB")
                    tensors[f"ref_image_{index}"] = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
                result = MiniMaxH3ReferenceToVideo.execute(clip, vae, audio_vae, prompt, width, height, length,
                    ref_image_size="match", ref_images=tensors).result
                refs = result[0][0][1].get("minimax_refs", [])
                assert len(refs) == 2 and all(ref["kind"] == "image" for ref in refs), "Missing reference conditioning"
                counts["reference_conditioning_blocks"] += len(refs)
                print("REFERENCE_CONDITIONING: two images encoded into minimax_refs in trainer, trainee order", flush=True)
                return result
        nodes.NODE_CLASS_MAPPINGS["BenchmarkReference"] = BenchmarkReference
        if args.attention == "sage":
            if not attention.SAGE_ATTENTION_IS_AVAILABLE:
                raise RuntimeError("SageAttention is unavailable")
            if importlib.metadata.version("sageattention") != "2.2.0":
                raise RuntimeError("Expected installed SageAttention 2.2.0")
            import sageattention.core as sage_core
            original_sage = attention.sageattn
            original_cuda_sage = sage_core.sageattn_qk_int8_pv_fp8_cuda
            def counted_sage(*values, **options):
                counts["sage_attempts"] += 1
                try:
                    result = original_sage(*values, **options)
                    counts["sage_completed"] += 1
                    return result
                except Exception:
                    counts["sage_errors"] += 1
                    raise
            def counted_cuda_sage(*values, **options):
                result = original_cuda_sage(*values, **options)
                counts["sage_qk_int8_pv_fp8_cuda_completed"] += 1
                return result
            attention.sageattn = counted_sage
            sage_core.sageattn_qk_int8_pv_fp8_cuda = counted_cuda_sage
            class BenchmarkSage:
                @classmethod
                def INPUT_TYPES(cls):
                    return {"required": {"model": ("MODEL",)}}
                RETURN_TYPES = ("MODEL",)
                FUNCTION = "patch"
                CATEGORY = "benchmark"
                def patch(self, model):
                    clone = model.clone()
                    def override(original, *values, **options):
                        counts["sage_dispatches"] += 1
                        completed_before = counts["sage_completed"]
                        result = attention.attention_sage(*values, **options)
                        if counts["sage_completed"] == completed_before:
                            counts["sage_pytorch_fallbacks"] += 1
                        return result
                    clone.model_options["transformer_options"]["optimized_attention_override"] = override
                    return (clone,)
            nodes.NODE_CLASS_MAPPINGS["BenchmarkSage"] = BenchmarkSage
        from comfy.ldm.minimax.model import MiniMaxH3Model
        original_forward = MiniMaxH3Model.forward
        def counted_forward(self, *a, **kw):
            pruned = bool(getattr(self, "use_adaln_curves", False))
            if pruned != report["base_model"]["pruned"]:
                raise RuntimeError("Loaded base pruning mode differs from requested base")
            counts["pruned_transformer_forwards" if pruned else "full_transformer_forwards"] += 1
            counts["transformer_forwards"] += 1
            payload = kw.get("minimax_payload") or (a[4] if len(a) > 4 else {})
            assert payload and len(payload.get("refs", [])) == 2, "Reference did not reach transformer"
            counts["reference_conditioned_forwards"] += 1
            label = "DiT_step_" + str(counts["transformer_forwards"])
            torch.cuda.nvtx.range_push(label)
            step_start = time.perf_counter_ns()
            try:
                return original_forward(self, *a, **kw)
            finally:
                torch.cuda.nvtx.range_pop()
                meter.calls.append({"component": label, "start_ns": step_start, "end_ns": time.perf_counter_ns(), "seconds": (time.perf_counter_ns()-step_start)/1e9})
        MiniMaxH3Model.forward = counted_forward
        import comfy.model_management
        report["attention"] = attention.optimized_attention.__name__
        report["dynamic_vram"] = comfy.memory_management.aimdo_enabled
        if report["attention"] != "attention_pytorch" or not report["dynamic_vram"]:
            raise RuntimeError("PyTorch global attention and dynamic VRAM are required")
        report["attention_backend"] = args.attention
        report["attention_scope"] = "diffusion model only; text encoder and VAEs retain PyTorch attention" if args.attention == "sage" else "PyTorch globally"
        if args.attention == "sage":
            report["attention"] = "sageattention_2.2.0_dit_override"
            report["sageattention_version"] = importlib.metadata.version("sageattention")
            report["gpu_compute_capability"] = list(torch.cuda.get_device_capability())
        report["adapter"] = {"repo": "larryvrh/MiniMax-H3-Turbo-Lora", "revision": LORA_REVISION,
            "file": MODELS["loras"][0], "strength": 1.0, "low_vram": False,
            "sha256": hashlib.sha256((args.model_root / "loras" / MODELS["loras"][0]).read_bytes()).hexdigest()}
        report["turbo_commit"] = TURBO_COMMIT
        report["steps"] = args.steps
        report["scheduler"] = "simple"
        report["sampler"] = "MiniMaxH3TurboSampler"
        report["prompt"] = prompt
        report["references"] = [{"file": name, "picture_index": index, "role": role,
            "sha256": hashlib.sha256((ROOT / "nsight_reference/inputs" / name).read_bytes()).hexdigest(),
            "url": "https://s3.renderplatform.com/agent-assets/" + name} for index, (name, role) in enumerate([("trainer.png", "trainer"), ("trainee.png", "trainee")], 1)]
        report["requested"] = {"seed": -1, "width": 720, "height": 1280, "frames": 241, "fps": 24}
        report["reference_sizing"] = "match"
        report["export_plan"] = "Center crop 736 to 720 (8 pixels per side); retain first 241 frames and matching audio; export time excluded from native generation timing"

        report["compatibility_status"] = "Experimental Larryvrh v4-600 on ref2va; runtime success is not a quality guarantee"
        executor = execution.PromptExecutor(
            Progress(), cache_type=execution.CacheType.NONE,
            cache_args={"ram": 0, "ram_inactive": 0, "lru": 0},
        )
        torch.cuda.synchronize()
        report["initialization_seconds"] = time.perf_counter() - start
        meter = Measurements(torch)
        for kind in {n["class_type"] for n in workflow.values()}:
            cls = nodes.NODE_CLASS_MAPPINGS[kind]
            meter.wrap(cls, "execute" if "execute" in cls.__dict__ else cls.FUNCTION, "node/" + kind)
        meter.wrap(comfy.sd.CLIP, "tokenize", "text_tokenization")
        meter.wrap(comfy.sd.CLIP, "encode_from_tokens_scheduled", "text_encoder")
        meter.wrap(comfy.samplers.KSAMPLER, "sample", "denoising_loop")
        for index in range(args.runs + 1):
            phase = "warmup" if index == 0 else f"measured_{index}"
            workflow = graph(prompt, args.seed, phase, args.steps, args.attention)
            save(out / (phase + "_workflow.json"), workflow)
            valid, error, outputs, details = asyncio.run(execution.validate_prompt(phase, workflow, None))
            if not valid:
                raise RuntimeError(f"Invalid workflow: {error}; {details}")
            meter.calls.clear()
            counts.clear()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            report["status"] = phase
            save(out / "report.json", report)
            print(f"START {phase}", flush=True)
            stop = threading.Event()
            samples = []
            def monitor():
                while not stop.is_set():
                    try:
                        row = subprocess.check_output(["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu,power.draw", "--format=csv,noheader,nounits"], text=True).strip()
                        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
                        samples.append({"gpu": row, "compute_pids": processes})
                    except Exception as exc:
                        samples.append({"error": repr(exc)})
                    stop.wait(1)
            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()
            measured = index > 0
            telemetry_process = None
            if measured:
                telemetry_stop = out / "telemetry.stop"
                telemetry_process = None
                time.sleep(0.25)
                torch.cuda.cudart().cudaProfilerStart()
                torch.cuda.nvtx.range_push("measured_run")
            capture_start_ns = time.perf_counter_ns()
            profiler_context = contextlib.nullcontext()

            try:
                start = time.perf_counter()
                with profiler_context:
                    executor.execute(workflow, phase, execute_outputs=outputs)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                generation_end_ns = time.perf_counter_ns()
                export_seconds = None
                if measured and executor.success:
                    native = next((out / "runtime/output" / phase).glob("*.mp4"))
                    export_start = time.perf_counter_ns()
                    torch.cuda.nvtx.range_push("final_export")
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(native), "-map", "0:v:0", "-map", "0:a:0", "-vf", "crop=720:1280:8:0,trim=end_frame=241,setpts=PTS-STARTPTS", "-af", "atrim=duration=10.041666666666666,asetpts=PTS-STARTPTS", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-map_metadata", "-1", str(out / "measured.mp4")], check=True)
                    torch.cuda.nvtx.range_pop()
                    export_seconds = (time.perf_counter_ns()-export_start)/1e9
                    meter.calls.append({"component":"final_export", "start_ns":export_start,"end_ns":time.perf_counter_ns(),"seconds":export_seconds})
            finally:
                capture_end_ns = time.perf_counter_ns()
                if measured:
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.cudart().cudaProfilerStop()
                    telemetry_stop.touch()
                    if telemetry_process is not None:
                        telemetry_process.wait(timeout=20)
                stop.set()
                monitor_thread.join()
                save(out / (phase + "_gpu_samples.json"), samples)
            unexpected = {int(p) for sample in samples for p in sample.get("compute_pids", "").splitlines() if p.strip() and int(p) != os.getpid()}
            if unexpected:
                raise RuntimeError(f"GPU contention invalidates benchmark: {unexpected}")
            if not executor.success:
                raise RuntimeError(str(executor.status_messages))
            if counts.get("transformer_forwards") != args.steps:
                raise RuntimeError(f"Expected {args.steps} transformer forwards: {counts}")
            if args.attention == "sage" and counts.get("sage_qk_int8_pv_fp8_cuda_completed", 0) <= 0:
                raise RuntimeError(f"Sage CUDA path did not execute: {counts}")
            videos = list((out / "runtime/output" / phase).glob("*.mp4"))
            if len(videos) != 1:
                raise RuntimeError(f"Expected one video, got {videos}")
            video = videos[0]
            probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)], text=True))
            stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
            if (stream["width"], stream["height"], int(stream["nb_frames"]), Fraction(stream["avg_frame_rate"])) != (736, 1280, 243, 24):
                raise RuntimeError(f"Unexpected output dimensions or timing: {stream}")
            if not any(s["codec_type"] == "audio" for s in probe["streams"]):
                raise RuntimeError("Generated audio is missing")
            totals = Counter()
            for call in meter.calls:
                totals[call["component"]] += call["seconds"]
            run = {"phase": phase, "generation_seconds": elapsed, "components_seconds": dict(totals),
                   "calls": list(meter.calls), "forward_counts": dict(counts),
                   "capture_start_ns":capture_start_ns,"capture_end_ns":capture_end_ns,"generation_end_ns":generation_end_ns,"export_seconds":export_seconds,
                   "sampled_peak_device_memory_mib": max((int(x["gpu"].split(",")[1]) for x in samples if "gpu" in x), default=None),
                   "gpu_monitor_interval_seconds": 1,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                   "video": str(video), "ffprobe": probe}
            if measured:
                report["runs"].append(run)
            save(out / "report.json", report)
            print(f"DONE {phase}: {elapsed:.3f} seconds; {video}", flush=True)
            comfy_main.hook_breaker_ac10a0.restore_functions()
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        if meter:
            report["partial_calls"] = meter.calls
        raise
    finally:
        if meter:
            meter.close()
        save(out / "report.json", report)


if __name__ == "__main__":
    main()
