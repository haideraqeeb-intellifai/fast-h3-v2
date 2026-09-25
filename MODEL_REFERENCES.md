# Model references — weights are not distributed here

## FastH3 V2 used by the 183.368 s and 293.715 s runs

- Model: [FastVideo/FastVideo-FastH3-8-Step-V2](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2/tree/3da2ddfe1954d9cda4c05b643dc0f26007a655c5).
- Exact revision: `3da2ddfe1954d9cda4c05b643dc0f26007a655c5`.
- Purpose: distilled text-to-audio-video transformer, text encoder, video VAE and audio VAE.
- License: MiniMax H3 Community License; Qwen-related terms and notices are retained in `models/FastH3-8-Step-V2/LICENSE`, `LICENSE-Qwen3-VL` and `NOTICE`.
- Access: the saved Hub metadata reports an ungated public repository. Download and use are subject to its license terms; authenticate with the Hugging Face CLI if required by the host.
- `MODEL_FILES.json` records all 32 excluded weight shards, their byte sizes and SHA-256 values from the original Hugging Face download metadata. The original checkpoint verification log is retained in `work/checkpoint_verification.log`.
- The experimental four-step directory is a relative symlink view of the same weights, with only the inference schedule changed. It is not a separately trained four-step checkpoint.

From the repository root, after installing the environment:

```bash
.venv/bin/hf download FastVideo/FastVideo-FastH3-8-Step-V2 \
  --revision 3da2ddfe1954d9cda4c05b643dc0f26007a655c5 \
  --local-dir models/FastH3-8-Step-V2
.venv/bin/python prepare_four_step_model.py
```

The download is approximately 148 GB in decimal units. Weights must stay local: do not force-add them to Git or LFS.

## Additional profiling archive

`nsight-*` and `h3-ref2va-4step-nsight-reproduction/` preserve a separate ComfyUI Ref2VA profiling experiment. They are not the FastH3 V2 timings above. Their saved runner specifies its ComfyUI and adapter commits, model revisions and command-line paths; `nsight-work/model_checksums.json` preserves model file sizes and hashes. These scripts require the separate ComfyUI runtime and model storage described in the saved report, and are not installed by the FastVideo setup below.

- Base/encoder/VAEs: [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/a98869194787969724c7425d95d0ed73ce9202af), revision `a98869194787969724c7425d95d0ed73ce9202af`. The public ungated API was checked during publication and contains all four recorded base/encoder/VAE filenames. The model card specifies custom license terms; consult its license and notices before use.
- LoRA: [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/tree/43a74557ac3f6539db8e0f2a959d03feb7a81480), revision `43a74557ac3f6539db8e0f2a959d03feb7a81480`, file `minimax_h3_turbo_v4_step600_ema.safetensors`. Consult the adapter model card and inherited MiniMax terms.
- Download those specific files with `hf download REPO FILE --revision REVISION --local-dir MODEL_ROOT`. Preserve `diffusion_models/`, `text_encoders/`, `vae/` and `loras/` placement and pass the resulting path using the saved runner's `--model-root` option. Verify against `nsight-work/model_checksums.json`.
- Runtime: [ComfyUI](https://github.com/Comfy-Org/ComfyUI/tree/e377e263049f9338b4d12a3dd417b36ae62948ff) and [ComfyUI-MiniMax-H3-Turbo](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/tree/4274783a23afcfdbea3b4876cb79effd6c510785). Supply their checkout paths through `--comfy-root` and `--turbo-root`; inspect `nsight-work/nsys_command.json` for the archived invocation and platform-specific profiler setup.

No base weights, LoRA weights, optimizer states or checkpoint shards are included in this publication.
