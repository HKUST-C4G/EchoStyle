#!/bin/bash

# Training script for Wan models.
# Wan2.1 Usage: ./train.sh <config> <train_data_path> <validation_data_path> <output_dir>
# Wan2.2 Usage: ./train.sh <config> <train_data_path> <validation_data_path> <output_dir> <wan22_task> <wan22_train_part>

set -e

# Parse arguments.
config=$1
data_path=$2
validation_data_file=$3
output_dir=$4
cache_path=$6

# Default values.
master_port=${MASTER_PORT:-29500}

# Task-specific configurations.
case $config in
    "wan22_i2v_fused_480p")
        # if [ $# -lt 5 ]; then
        #     echo "Usage: $0 wan22_ti2v_fused_720p <train_data_path> <validation_data_path> <output_dir>" \
        #          "<wan22_train_part>"
        #     exit 1
        # fi
        pretrained="../models/Wan2.2_I2V_14B/"
        task="i2v-A14B"
        flow_shift=12
        max_pixels=230400  # 480p
        torchrun_args="--nnodes=\${WORLD_SIZE} --nproc_per_node=8 --rdzv_id=333 --rdzv_backend=c10d --rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT}"
        #torchrun_args="--nproc_per_node=1 --master_port $master_port"
        validation_args=""  # Validation is not supported for Wan2.2
        wan22_args="--wan22_train_part \"$5\""
        ;;
    *)
        echo "Error: Unknown config '$config'"
        echo "Available configs: wan21_i2v_480p, wan21_i2v_fused_720p, wan21_i2v_fused_720p_zj," \
             "wan21_flf2v_fused_720p, wan21_vace_fused_720p, wan22_fused_720p, wan22_flash_fused_720p"
        exit 1
        ;;
esac

# Set Wandb base URL.
export WANDB_BASE_URL=https://api.bandw.top

# Build the torchrun command.
cmd="WANDB_MODE=offline PYTHONPATH=. python -m torch.distributed.run $torchrun_args trainer/cli/train_wan2.2_i2v.py"

# Add common arguments
cmd="$cmd --pretrained \"$pretrained\""
cmd="$cmd --enable_lora"
#cmd="$cmd --enable_fsdp"
cmd="$cmd --lora_rank 64"
cmd="$cmd --lora_alpha 32"
cmd="$cmd --data_path \"$data_path\""
cmd="$cmd --cache_path \"$cache_path\""
cmd="$cmd --frame_bucket 33 37 41 45 49 53 57 61 65 69"
cmd="$cmd --train_fps 16"
cmd="$cmd --max_pixels $max_pixels"
cmd="$cmd --dataloader_num_workers 4"
cmd="$cmd --learning_rate 2e-5"
cmd="$cmd --train_batch_size 1"
cmd="$cmd --gradient_accumulation_steps 1"
cmd="$cmd --enable_gradient_checkpointing"
cmd="$cmd --max_grad_norm 1.0"
cmd="$cmd --max_train_steps 50000"
cmd="$cmd --weighting_scheme \"logit_normal\""
cmd="$cmd --flow_shift $flow_shift"
cmd="$cmd --output_dir \"training_outputs/$output_dir\""
cmd="$cmd --logging_dir \"logs\""
cmd="$cmd --report_to \"wandb\""
cmd="$cmd --checkpointing_steps 2000"
cmd="$cmd --resume_from_checkpoint \"latest\""
cmd="$cmd --validation_steps 50001"
cmd="$cmd --validation_start_step 0"
cmd="$cmd --spatial_downsample_factor 16"
cmd="$cmd --task \"$task\""
cmd="$cmd --seed 42"

# Add validation arguments if available.
if [ -n "$validation_args" ]; then
    cmd="$cmd $validation_args"
fi

# Add Wan2.2 specific arguments if available.
if [ -n "$wan22_args" ]; then
    cmd="$cmd $wan22_args"
fi

echo "Executing: $cmd"
eval $cmd