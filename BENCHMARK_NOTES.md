# FastH3 8-Step V2 matched-input benchmark

Reference: https://s3.renderplatform.com/agent-assets/Raqeeb/larryvrh-h3-v4-600-4steps/index.html

Model: `FastVideo/FastVideo-FastH3-8-Step-V2`, revision
`3da2ddfe1954d9cda4c05b643dc0f26007a655c5`.

The runner uses the reference prompt (stripped terminal newline), seed 42,
736 × 1280, 243 frames, 24 FPS, and guidance 1.0. It runs one warm-up and one
measured request in the same process, with seed 42 for both. The model's trained
eight-forward schedule, shifts 10/3, VSA-H3 80% sparsity and 64-token tiles are
preserved. On the RTX 5090 it uses the Triton VSA kernel, BF16 weights, CPU
offloading with sequential text-encoder release, and no explicit DiT/VAE compilation. Official H3 inference fusions are enabled.

```bash
OMP_NUM_THREADS=8 .venv/bin/python -u benchmark.py > run/run.log 2>&1
.venv/bin/python build_report.py
```

`work/cpu-load.patch` fixes a load-time VRAM overflow in this pinned FastVideo
checkout: for layerwise offloading, it loads transformer weights onto CPU,
attaches the existing offload hooks, and moves remaining parameters/buffers to
CUDA. This does not change weights, tensor precision, attention, or sampling.
The initial failed initialization is retained in `work/failed-load/`.

`run/report.json` records runtime configuration, timings and stage metrics.
Each run has per-second GPU samples and its exact request. Checkpoint verification
is recorded in `work/checkpoint_verification.log`. `build_report.py` checks the
video dimensions and frame count, completely decodes both audio/video streams,
compares decoded hashes, and assembles `fasth3-8-step-v2-seed42/index.html`.

Generation time includes the synchronous request through final video/audio
saving. Initialization, downloads, checks and report construction are separately
reported. One measured sample does not establish timing variance. Different
runtime, precision and attention choices mean this is a comparison of complete
configurations, not an isolated checkpoint-speed comparison.

The resident-model attempt saved a valid warm-up but its worker exited during
measured denoising (EOFError; host-memory pressure suspected). It is retained
in `work/failed-resident-run/`. The final runner uses `h3_sequential_load=True`
and optional CPU pinning disabled. Model reloads inside a request count toward
its full generation time. Host available memory is included in per-second samples.

The unfused sequential attempt exceeded GPU memory in SwiGLU and is preserved
in `work/failed-sequential-unfused/`. The final configuration uses the official
`all` H3 fusion profile and `PYTORCH_ALLOC_CONF=expandable_segments:True`.
Fusions can change floating-point operation order. `run_when_idle.py` waits
for other GPU compute jobs before launching the benchmark.
