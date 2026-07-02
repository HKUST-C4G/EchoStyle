#!/bin/bash
# =============================================================================
# Wan2.2 14B I2V Training Launcher
# =============================================================================
# Quick-start wrapper that invokes the main training script (train_wan22_i2v.sh)
# with pre-filled arguments for the Wan2.2 14B image-to-video task.
#
# This trains a LoRA adapter on the Wan2.2 I2V 14B DiT model at 480p resolution.
# The Wan2.2 dual-LoRA scheme splits training by timestep range:
#   - "high_noise": trains on timesteps t=900–1000 (coarse structure)
#   - "low_noise":  trains on timesteps t=0–900   (detail refinement)
# Run this script once per train_part to produce both LoRA checkpoints.
#
# Arguments passed to train_wan22_i2v.sh:
#   $1 = config name         : "wan22_i2v_fused_480p" — 14B model at 480p resolution
#   $2 = training data       : JSON file with video paths and prompts
#   $3 = validation data     : JSON file for validation (not used in Wan2.2)
#   $4 = output name         : subdirectory name under training_outputs/
#   $5 = train_part           : "low_noise" or "high_noise" — which timestep range to train
#   $6 = cache_path           : directory for pre-encoded VAE/T5 latent caches
#
# Usage:
#   bash scripts/wan2.2_i2v.sh
#
# To train both LoRA models, run twice with different train_part:
#   # Edit $5 to "high_noise", then run again
# =============================================================================

bash scripts/train_wan22_i2v.sh wan22_i2v_fused_480p \
../datasets/Anime20k/metadata.json \
scripts/vace_bench.json echostyle low_noise \
../datasets/Anime20k/480p_cache
