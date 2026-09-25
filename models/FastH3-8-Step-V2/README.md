---
license: other
license_name: minimax-h3-community
license_link: LICENSE
base_model: MiniMaxAI/MiniMax-H3
library_name: diffusers
pipeline_tag: text-to-video
tags:
- text-to-video
- video
- audio
- text-to-audio-video
- distillation
- dmd2
- few-step
- minimax-h3
- fastvideo
- fasth3
---

<p align="center">
  <a href="https://github.com/hao-ai-lab/FastVideo"><img src="https://raw.githubusercontent.com/hao-ai-lab/FastVideo/main/assets/logos/logo.svg" width="320" alt="FastVideo"></a>
</p>

# FastVideo-FastH3-8-Step-V2

The FastH3 8-Step V2 checkpoint from
[FastVideo](https://github.com/hao-ai-lab/FastVideo). It generates synchronized
video and audio from text with eight transformer forwards. This step-1300 model
was trained with data-free DMD2 and VSA-H3 at 80% sparsity.

[Blog](https://haoailab.com/blogs/fasth3-preview/) ·
[FastH3 collection](https://huggingface.co/collections/FastVideo/fastvideo-fasth3)

> This checkpoint requires FastVideo's VSA-H3 attention backend. Its video
> scheduler shift is 10, not the base model's 12; use the example below, which
> reads the trained schedule from the checkpoint.

## Run with FastVideo

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then use
the CUDA 13 / Blackwell path below. It selects FastVideo's published CUDA
kernel wheel instead of compiling the kernel locally. See the
[installation guide](https://hao-ai-lab.github.io/FastVideo/getting_started/installation/)
for other platforms.

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo
uv venv --python 3.12 --seed
source .venv/bin/activate
UV_TORCH_BACKEND=cu130 uv pip install \
  --no-sources-package fastvideo-kernel \
  -e ".[fasth3]"
```

```bash
python examples/inference/basic/basic_fasth3_8step.py \
  --prompt "your prompt" \
  --no-warmup \
  --repeats 1
```

The tested defaults use four B200 GPUs and the trained eight-forward schedule.
On other multi-GPU CUDA systems, follow the installation guide and add
`--no-replicated-dit --vsa-kernel triton --no-fa4`. The GPU count must divide
H3's 56 attention heads.

## Scope

This checkpoint supports text-to-audio-video generation. FL2VA and Ref2VA were
not distilled. Difficult motion, fine detail, and some audio may remain below
the base MiniMax H3 model. This checkpoint inherits the
[MiniMax H3 Community License](LICENSE).

## Acknowledgements

We thank [Nuva Lab](https://nuvalab.ai/) for bringing production grounding to FastH3 through its experience with real-world creative video-agent workloads. Its production-aligned post-training insights help bridge open-source research to practical data-assisted distillation for commercial video workflows, with Omni Ref as the next focus.

We thank the [NVIDIA FastGen](https://github.com/NVlabs/FastGen) team for the [DMD2](https://arxiv.org/abs/2405.14867) framework and H3 reference experiment that helped us align the score clock, modality shifts, and backward simulation.

We also thank [MiniMax](https://huggingface.co/MiniMaxAI/MiniMax-H3) for releasing H3-Base, and the [vLLM project](https://vllm.ai/), [NVIDIA](https://www.nvidia.com/en-us/), and [MBZUAI](https://mbzuai.ac.ae/) for their continued sponsorship and support of [FastVideo](https://github.com/hao-ai-lab/FastVideo).
