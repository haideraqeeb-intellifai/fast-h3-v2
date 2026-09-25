# FastH3 V2 — RTX 5090 benchmark configurations

Code, patches, model metadata, generated videos and measurement evidence for two matched-input FastH3 V2 runs:

| Configuration | Measured full generation | Denoising | Saved run |
|---|---:|---:|---|
| Experimental four forwards, decoder copy-back removed | **183.368 s** | 106.274 s | `run_4step_no_copyback/` |
| Trained eight forwards, original decoder copy-back | **293.715 s** | 213.769 s | `run/` |

Both generated 243 frames at 736×1280 and 24 FPS with audio, using seed 42 on an RTX 5090. Each timing follows a warm-up. Full generation includes request-time model loading, offloading, encoding, denoising, decoding and saving; initialization is separate. These are individual observations. The comparison changes both schedule and decoder policy and does not establish equal quality.

- [Reproduce either configuration](REPRODUCE.md)
- [Model download references and licenses](MODEL_REFERENCES.md) — **no model weights are included**
- [Four-step report](fasth3-v2-experimental-4-step-seed42/index.html) and [measured video](run_4step_no_copyback/measured.mp4)
- [Eight-step report](fasth3-8-step-v2-seed42/index.html) and [measured video](run/measured.mp4)
- [Original benchmark notes](BENCHMARK_NOTES.md)
- [Publication inventory](PUBLICATION_INVENTORY.json) and [publication log](PUBLISHING_LOG.md)

The repository includes the modified FastVideo source under `work/FastVideo/`, pinned upstream provenance, both patches, runners, report builders, package inventory, prompt, warm-up and measured media, telemetry and earlier attempts. The original four-step run with decoder copy-back (185.787 s) is preserved for comparison. Additional `nsight-*` directories and `h3-ref2va-4step-nsight-reproduction/` retain a separate ComfyUI Ref2VA profiling archive, not a FastH3 V2 result.

Large profiling files use Git LFS. Run `git lfs pull` after cloning. To view the HTML reports, serve the clone with `python3 -m http.server 8000`; GitHub's file view does not render them as a website.

FastVideo's Apache 2.0 license and bundled third-party notices are retained beside its source. Model metadata retains the upstream MiniMax H3 Community License and Qwen notices. No new license is assigned to the benchmark-specific work.
