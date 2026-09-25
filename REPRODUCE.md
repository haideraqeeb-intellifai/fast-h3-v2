# Reproduce the FastH3 V2 configurations

The recorded environment was Linux, Python 3.12.13, PyTorch 2.12.0+cu130, Triton 3.7.0 and an NVIDIA RTX 5090 with 32 GB VRAM. The full installed package list is `work/packages.txt`; its editable-package path is historical. Model weights occupy approximately 148 GB on disk. CPU offloading needs substantial host RAM; see `work/hardware.json` and the saved memory samples. Do not infer sufficient RAM from GPU capacity alone.

## Get the publication and install

```bash
git lfs install
git clone https://github.com/haideraqeeb-intellifai/fast-h3-v2.git
cd fast-h3-v2
git lfs pull
uv venv --python 3.12 --seed
UV_TORCH_BACKEND=cu130 uv pip install --python .venv/bin/python \
  --no-sources-package fastvideo-kernel -e './work/FastVideo[fasth3]'
```

Install system `ffmpeg`/`ffprobe` for media checks and report building. The published FastVideo kernel wheel is used. Optional upstream submodules are pinned in the root `.gitmodules` and can be fetched with `git submodule update --init --recursive` for source kernel builds or VBench evaluation; they are unnecessary for these inference commands.

Download the pinned model using [MODEL_REFERENCES.md](MODEL_REFERENCES.md). The vendored FastVideo source is pinned in `SOURCE_PROVENANCE.json`; both local patches are already applied. Do not apply `cpu-load.patch` a second time.

## Experimental four-step run — recorded 183.368 s

```bash
.venv/bin/python prepare_four_step_model.py
.venv/bin/python publication_runtime.py no_copyback
OMP_NUM_THREADS=8 PYTORCH_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u benchmark_4step_no_copyback.py > run_4step_no_copyback/run.log 2>&1
.venv/bin/python build_report_4step_no_copyback.py
```

This uses four transformer forwards at `[999, 749, 500, 250]`. The runner passes `--steps 5` because the checkpoint contract includes a terminal step; the log records four forwards. Decoder CPU weight storage is retained while GPU copies are discarded after decode.

## Trained eight-step run — recorded 293.715 s

```bash
.venv/bin/python publication_runtime.py copyback
OMP_NUM_THREADS=8 PYTORCH_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u benchmark.py > run/run.log 2>&1
.venv/bin/python build_report.py
```

This restores the original decoder copy-back behavior and uses eight transformer forwards at `[999, 874, 749, 624, 500, 375, 250, 125]`. Switch back with `publication_runtime.py no_copyback` before running the four-step no-copyback benchmark again. Do not switch variants while any benchmark is running. The root runners check the decoder and CPU-loader source hashes before starting.

Both runs use the same prompt, seed 42, 736×1280 pixels, 243 frames, 24 FPS, guidance 1.0, BF16, Triton VSA-H3 with 80% sparsity and 64-token tiles, CPU layerwise DiT offload, sequential text-encoder release, official H3 fusions, and no explicit model/VAE compilation. Each script runs one warm-up and one measured request and refuses to start if another GPU compute process is present.

The commands overwrite the saved run artifacts; use a separate clone to preserve the published evidence. The no-copyback report builder also asserts parity with the saved original four-step video/audio hashes; a new run on another environment may fail this assertion even if it is valid. The original executed scripts, configs, hashes and logs remain in each `run*/` directory and the report assets. Root entrypoints only add publication portability and configuration checks.

## Inspect without generating

The saved reports are `fasth3-8-step-v2-seed42/index.html` and `fasth3-v2-experimental-4-step-seed42/index.html`. Serve the repository with `python3 -m http.server 8000` and open these paths in a browser. Reports contain embedded local video links, configuration data, timings and verification results. Existing saved JSON retains original machine paths as provenance; root report builders resolve the media from the local clone.

These timings are individual observations, not latency guarantees or an equal-quality comparison. The four-step schedule is an unvalidated subsampling of an eight-step distilled checkpoint. The eight-step run also used a different decoder policy. FastH3 V2 did not distill Ref2VA; the separate Nsight archive uses a different ComfyUI configuration.
