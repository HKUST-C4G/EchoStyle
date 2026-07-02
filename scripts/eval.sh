#!/bin/bash
# =============================================================================
# Wan2.2 Evaluation Script — Video-to-Video Style Transfer
# =============================================================================
# This script runs inference/evaluation using trained LoRA weights on the
# Wan2.2 DiT model. It supports both the 5B (TI2V) and 14B (I2V) variants.
#
# The Wan2.2 architecture uses a dual-LoRA scheme:
#   - high_lora_path: LoRA weights for high-noise timesteps (t=900–1000)
#   - low_lora_path:  LoRA weights for low-noise timesteps  (t=0–900)
# Both are applied at their respective timestep ranges during sampling.
#
# Key parameters:
#   --dit_fsdp / --t5_fsdp   : Enable FSDP sharding for DiT / T5 to fit in GPU memory
#   --ulysses_size            : Number of GPUs for Ulysses sequence parallelism
#   --sample_guide_scale      : Classifier-free guidance scale (higher = stronger style)
#   --concat True             : Concatenate source video with generated output
#   --sample_steps            : Number of denoising steps (fewer = faster, lower quality)
#   --frame_num               : Number of output frames to generate
#
# Usage:
#   bash scripts/eval_2.2_5b_v2v.sh
#
# Prerequisites:
#   - Pretrained model weights in ../models/
#   - Trained LoRA checkpoints (replace placeholder paths below)
#   - PYTHONPATH=. is required for module resolution
# =============================================================================

# -----------------------------------------------------------------------------
# [DISABLED] Wan2.2 5B TI2V evaluation — basic configuration
# Uses 4 GPUs with FSDP, 1280x704 resolution, default 33 frames
# -----------------------------------------------------------------------------
# PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=4 trainer/models/wan22/eval_2.25b.py \
#  --task ti2v-5B --size 1280*704  --dit_fsdp --t5_fsdp --ulysses_size 4 --ring_size 1 \
#  --ckpt_dir ../models/Wan2.2-TI2V/ \
#  --data_path ../datasets/cartoon20k/720p_1/test_sample/test_data.json \
#  --output_dir results/wan2.2_10000_tv_cfg \
#  --lora_path training_outputs/echostyle_wan2.2_5b_lora_64_fused480p/checkpoint-10000/model_state.pth

# -----------------------------------------------------------------------------
# [DISABLED] Wan2.2 5B TI2V evaluation — extended frames + guidance tuning
# Generates 61 frames with guidance scale 6.0
# -----------------------------------------------------------------------------
# PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=4 trainer/models/wan22/eval_2.25b.py \
#  --task ti2v-5B --size 1280*704 --ulysses_size 4 --ring_size 1 \
#  --ckpt_dir ../models/Wan2.2-TI2V/ \
#  --data_path ../datasets/cartoon20k/720p_1/test_sample/test_data.json \
#  --output_dir results/wan2.2_5b_10000_61f \
#  --lora_path training_outputs/echostyle_wan2.25b_64lora_1226/checkpoint-10000/model_state.pth \
#  --frame_num 61 \
#  --sample_guide_scale 6.0 --concat True

# -----------------------------------------------------------------------------
# [ACTIVE] Wan2.2 14B I2V evaluation — standard model with dual LoRA
# Uses 8 GPUs, 1280x720 (720p), guidance scale 5.0
# NOTE: Replace high_noise_lora_path / low_noise_lora_path with actual checkpoint paths
# -----------------------------------------------------------------------------
PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=8 trainer/models/wan22/eval.py \
 --task i2v-A14B --size 1280*720 --dit_fsdp --t5_fsdp --ulysses_size 8 --ring_size 1 \
 --ckpt_dir ../models/Wan2.2_I2V_14B/ \
 --data_path scripts/test_10.json \
 --output_dir results/demo \
 --high_lora_path high_noise_lora_path \
 --low_lora_path low_noise_lora_path \
 --sample_guide_scale 5.0 --concat True

# -----------------------------------------------------------------------------
# [ACTIVE] Wan2.2 14B I2V evaluation — lightweight/distilled model variant
# Uses the "light" checkpoint with lower guidance (1.0) and fewer steps (10)
# for faster inference at the cost of some quality
# NOTE: Replace high_noise_lora_path / low_noise_lora_path with actual checkpoint paths
# -----------------------------------------------------------------------------
 PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=8 trainer/models/wan22/eval.py \
 --task i2v-A14B --size 1280*720 --dit_fsdp --t5_fsdp --ulysses_size 8 --ring_size 1 \
 --ckpt_dir ../models/wan2.2_i2v_light/ \
 --data_path scripts/test_10.json \
 --output_dir results/demo_light \
 --high_lora_path high_noise_lora_path \
 --low_lora_path low_noise_lora_path \
 --sample_guide_scale 1.0 --concat True --sample_steps 10

# -----------------------------------------------------------------------------
# [DISABLED] Wan2.2 14B I2V evaluation — single-GPU, multitask LoRA
# Runs without FSDP on a single GPU (for debugging or small-scale testing)
# -----------------------------------------------------------------------------
# PYTHONPATH=. python trainer/models/wan22/eval.py \
#  --task i2v-A14B --size 1280*720 \
#  --ckpt_dir ../models/Wan2.2_I2V_14B/ \
#  --data_path ../datasets/cartoon20k/720p_1/test_sample/test_data.json \
#  --output_dir results/wan-i2v-14b-multi-4000 \
#  --high_lora_path training_outputs/wan2.2_i2v_high_noise_multitask/checkpoint-4000/model_state.pth \
#  --low_lora_path training_outputs/wan2.2_i2v_low_noise_multitask/checkpoint-4000/model_state.pth \
#  --sample_guide_scale 6.0 --concat True
