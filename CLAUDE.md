# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

wan_trainer is a distributed training and evaluation framework for Alibaba's Wan series video generation models (Wan2.1 and Wan2.2). It fine-tunes diffusion transformer (DiT) models for video-to-video (v2v) and image-to-video (i2v) tasks using LoRA, with support for FSDP and sequence parallelism across multi-node GPU clusters.

## Commands

### Install
```bash
pip install -e .
```

### Training (multi-GPU, via torchrun)
```bash
# Full command (the shell script wraps torchrun):
bash scripts/train_wan22_i2v.sh wan22_i2v_fused_480p \
  <data_json> <validation_json> <output_name> <train_part> <cache_path>

# Example:
bash scripts/wan2.2_i2v.sh   # pre-filled example invocation

# Direct torchrun (for debugging on fewer GPUs):
PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=1 \
  trainer/cli/train_wan2.2_i2v.py \
  --pretrained ../models/Wan2.2_I2V_14B/ \
  --task i2v-A14B --enable_lora --lora_rank 64 \
  --data_path <data.json> --cache_path <cache_dir> \
  --output_dir training_outputs/<name> \
  --enable_gradient_checkpointing --max_train_steps 50000
```

### Evaluation
```bash
# Wan2.2 14B eval with dual LoRA (high-noise + low-noise):
PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=8 \
  trainer/models/wan22/eval.py \
  --task i2v-A14B --size 1280*720 --dit_fsdp --t5_fsdp \
  --ulysses_size 8 --ring_size 1 \
  --ckpt_dir ../models/Wan2.2_I2V_14B/ \
  --data_path <test.json> --output_dir results/<name> \
  --high_lora_path <high_noise_ckpt>/model_state.pth \
  --low_lora_path <low_noise_ckpt>/model_state.pth
```

### Lint
```bash
ruff check .
ruff format --check .
```

## Architecture

### Training Pipeline (`trainer/cli/train_wan2.2_i2v.py`)

The main training script follows this flow:

1. **Init distributed** — Sets up data-parallel + sequence-parallel process groups via `trainer.utils.distributed`. FSDP uses `HYBRID_SHARD` with the device mesh `(dp, sp)`.
2. **Load models** — VAE (`Wan2_1_VAE`), T5 text encoder, and DiT (`WanModel`) loaded from pretrained weights. CLIP is only used for Wan2.1 tasks.
3. **LoRA injection** — When `--enable_lora`, all `nn.Linear` modules in the DiT get LoRA adapters via HuggingFace PEFT. The base model is frozen in bf16; LoRA params remain fp32 (unless FSDP, which forces uniform bf16).
4. **Dataset caching** — `BaseDataset.calculate_or_load_dataset_cache()` pre-encodes all videos through VAE and text encoder, saving `.safetensors` files to `--cache_path`. Subsequent runs load from cache directly.
5. **Training loop** — Flow-matching loss with configurable timestep weighting (`logit_normal`, `cosmap`, etc.). Wan2.2 splits training into timestep ranges (`high_noise` / `low_noise`) for separate LoRA models.
6. **Checkpointing** — Saves `model_state.pth` + `optimizer_state.pth` under `checkpoint-{step}` dirs. FSDP uses `FullStateDictConfig` for rank-0-only saves.

### Wan2.2 Dual-LoRA Architecture

Wan2.2 trains **two separate LoRA models** for different timestep ranges:
- `high_noise` (t=900–1000): Handles heavily noised inputs
- `low_noise` (t=0–900): Handles refinement/denoising

At evaluation time, both LoRAs are loaded and applied at their respective timestep ranges during the sampling process (see `eval.py --high_lora_path` / `--low_lora_path`).

### Key Modules

- **`trainer/models/wan22/wan/modules/model.py`** — `WanModel` (DiT), extends `diffusers.ModelMixin`. Supports sequence parallelism and gradient checkpointing.
- **`trainer/models/wan22/wan/modules/attention.py`** — Flash attention implementation with Ulysses sequence parallelism.
- **`trainer/models/wan22/wan/configs/`** — Model configs (`WAN_CONFIGS` dict) mapping task names to hyperparameters.
- **`trainer/datasets/base_dataset.py`** — Base class handling video loading (via decord), frame bucketing, FPS resampling, resolution scaling, and distributed cache computation.
- **`trainer/datasets/wanv2v_dataset.py`** — V2V dataset: loads paired (source, target) videos. Caches video latents, source latents, and text embeddings.
- **`trainer/utils/distributed.py`** — Process group management via `init_device_mesh`. Provides data-parallel and sequence-parallel group accessors.
- **`trainer/utils/checkpoint.py`** — Save/load for both DDP and FSDP. `load_lora()` handles loading LoRA weights from `.pth` or safetensors format.

### Data Format

Training data is a JSON file: `[{"video_path": "...", "src_video_path": "...", "prompt": "..."}, ...]`
- `video_path`: target video
- `src_video_path`: source/reference video (for v2v tasks)
- `prompt`: text description (Chinese or English)

### Supported Tasks

Defined in `trainer/models/wan22/wan/configs/__init__.py`:
- `t2v-A14B`: text-to-video
- `i2v-A14B`: image-to-video (14B params)
- `ti2v-5B`: text+image-to-video (5B params)
- `flf2v-A14B`: first-last-frame to video
- `kf2v-A14B`: key-frame to video
- Flash variants (`*-flash`): optimized inference versions

## Important Conventions

- **PYTHONPATH=.** is required when running scripts directly (not via pip install).
- **decord must be imported after torch** to avoid segfaults (see `base_dataset.py`).
- Training always uses `batch_size=1` per device — video sizes vary, so no batching across samples.
- VAE is always fp32; T5 is bf16; DiT runs under `torch.amp.autocast("cuda", dtype=torch.bfloat16)`.
- Ruff config: line length 119, enforces import sorting (`I`) and unused imports (`F401`).
