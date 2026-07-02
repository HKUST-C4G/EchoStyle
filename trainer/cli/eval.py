import argparse
import glob
import os
import re

import torch


def find_model_at_specified_steps(model_name: str, steps: int | str):
    """
    Find the path of the specified model at the specified steps.

    Args:
        model_name: Model name, e.g., "wantrainer_i2v_childhood_fused_720p"
        steps: Number of steps, e.g., 2200, or string "latest" for the latest checkpoint

    Returns:
        The full path of the corresponding checkpoint, or None if it does not exist
    """
    base_path = os.path.join("training_outputs", model_name)

    # Check if model directory exists.
    if not os.path.exists(base_path):
        print(f"Error: Model directory '{model_name}' does not exist")
        return None

    # Handle steps="latest" case, find the largest checkpoint.
    if steps == "latest":
        checkpoint_dirs = glob.glob(os.path.join(base_path, "checkpoint-*"))
        if not checkpoint_dirs:
            print(f"Error: No checkpoint directories found in {base_path}")
            return None

        # Extract checkpoint numbers and find the maximum.
        checkpoint_nums = []
        for dir_path in checkpoint_dirs:
            match = re.search(r"checkpoint-(\d+)$", dir_path)
            if match:
                checkpoint_nums.append(int(match.group(1)))

        if not checkpoint_nums:
            print(f"Error: No valid checkpoint directories found in {base_path}")
            return None

        steps = max(checkpoint_nums)
        checkpoint_path = os.path.join(base_path, f"checkpoint-{steps}", "model_state.pth")
    else:
        # Handle steps=int case.
        checkpoint_path = os.path.join(base_path, f"checkpoint-{steps}", "model_state.pth")

    # Check if checkpoint exists.
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} does not exist")
        return None

    return checkpoint_path, steps


def parse_args():
    parser = argparse.ArgumentParser(description="Call {wan21|wan22}/generate.py to evaluate the trained model.")

    # Wan2.1
    parser.add_argument("--model", type=str, default=None, help="Wan2.1 model name.")
    parser.add_argument("--steps", type=str, default="latest", help="Number of steps or 'latest'.")

    # Wan2.2
    parser.add_argument("--high_model", type=str, default=None, help="Wan2.2 high noise model name.")
    parser.add_argument("--high_steps", type=str, default="latest", help="Number of steps or 'latest'.")
    parser.add_argument("--low_model", type=str, default=None, help="Wan2.2 low noise model name.")
    parser.add_argument("--low_steps", type=str, default="latest", help="Number of steps or 'latest'.")

    # Wan2.2-Flash
    parser.add_argument("--high1_model", type=str, default=None, help="Wan2.2-Flash high_noise_1 model name.")
    parser.add_argument("--high1_steps", type=str, default="latest", help="Number of steps or 'latest'.")
    parser.add_argument("--high2_model", type=str, default=None, help="Wan2.2-Flash high_noise_2 model name.")
    parser.add_argument("--high2_steps", type=str, default="latest", help="Number of steps or 'latest'.")
    parser.add_argument("--low3_model", type=str, default=None, help="Wan2.2-Flash low_noise_3 model name.")
    parser.add_argument("--low3_steps", type=str, default="latest", help="Number of steps or 'latest'.")
    parser.add_argument("--low4_model", type=str, default=None, help="Wan2.2-Flash low_noise_4 model name.")
    parser.add_argument("--low4_steps", type=str, default="latest", help="Number of steps or 'latest'.")

    # Common
    parser.add_argument("--data_file", type=str, help="Data JSON file.")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory.")
    parser.add_argument("--base_model", default="720P", choices=["480P", "720P"], help="Base model.")
    parser.add_argument(
        "--eval_size", default="1280*720", choices=["512*512", "832*480", "1280*720"], help="Evaluation size."
    )
    parser.add_argument(
        "--task",
        default="i2v",
        choices=[
            "i2v-14B",
            "t2v-1.3B",
            "flf2v-14B",
            "vace-1.3B",
            "vace-14B",
            "t2v-A14B",
            "i2v-A14B",
            "lf2v-A14B",
            "flf2v-A14B",
            "kf2v-A14B",
            "i2v-A14B-flash",
            "lf2v-A14B-flash",
            "flf2v-A14B-flash",
            "kf2v-A14B-flash",
        ],
        help="Task type.",
    )
    parser.add_argument("--max_gen_num", type=int, default=-1, help="Maximum number of generations.")
    parser.add_argument("--pattern", type=str, default=None, help="Pattern to filter data.")
    return parser.parse_args()


def maybe_int(steps):
    return int(steps) if steps != "latest" else steps


def print_yellow(text):
    print(f"\033[1;33m{text}\033[0m")


def main():
    args = parse_args()

    # Get CUDA_VISIBLE_DEVICES from environment variable.
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_devices is None:
        # If the environment variable is not set, use all available GPUs.
        num_gpus = torch.cuda.device_count()
    else:
        # Use the number of GPUs specified in the environment variable.
        num_gpus = len(cuda_devices.split(","))

    # Convert steps to int.
    steps = maybe_int(args.steps)
    high_steps = maybe_int(args.high_steps)
    low_steps = maybe_int(args.low_steps)
    high1_steps = maybe_int(args.high1_steps)
    high2_steps = maybe_int(args.high2_steps)
    low3_steps = maybe_int(args.low3_steps)
    low4_steps = maybe_int(args.low4_steps)

    # Find the model path.
    ckpt = high_ckpt = low_ckpt = high1_ckpt = high2_ckpt = low3_ckpt = low4_ckpt = None
    if args.model:
        ckpt, steps = find_model_at_specified_steps(args.model, steps)
        print_yellow(f"Model path: {ckpt}")
    if args.high_model:
        high_ckpt, high_steps = find_model_at_specified_steps(args.high_model, high_steps)
        print_yellow(f"Wan2.2 high_noise model path: {high_ckpt}")
    if args.low_model:
        low_ckpt, low_steps = find_model_at_specified_steps(args.low_model, low_steps)
        print_yellow(f"Wan2.2 low_noise model path: {low_ckpt}")
    if args.high1_model:
        high1_ckpt, high1_steps = find_model_at_specified_steps(args.high1_model, high1_steps)
        print_yellow(f"Wan2.2-Flash high_noise_1 model path: {high1_ckpt}")
    if args.high2_model:
        high2_ckpt, high2_steps = find_model_at_specified_steps(args.high2_model, high2_steps)
        print_yellow(f"Wan2.2-Flash high_noise_2 model path: {high2_ckpt}")
    if args.low3_model:
        low3_ckpt, low3_steps = find_model_at_specified_steps(args.low3_model, low3_steps)
        print_yellow(f"Wan2.2-Flash low_noise_3 model path: {low3_ckpt}")
    if args.low4_model:
        low4_ckpt, low4_steps = find_model_at_specified_steps(args.low4_model, low4_steps)
        print_yellow(f"Wan2.2-Flash low_noise_4 model path: {low4_ckpt}")

    # Determine the pretrained model directory and flow shift.
    is_wan22 = "A14B" in args.task
    if args.base_model == "720P":
        shift = {
            "i2v-14B": 5,
            "flf2v-14B": 16,
            "vace-1.3B": 16,
            "vace-14B": 16,
            "t2v-A14B": 12,
            "i2v-A14B": 5,
            "lf2v-A14B": 5,
            "flf2v-A14B": 5,
            "kf2v-A14B": 5,
            "i2v-A14B-flash": 5,
            "lf2v-A14B-flash": 5,
            "flf2v-A14B-flash": 5,
            "kf2v-A14B-flash": 5,
        }[args.task]
        ckpt_dir = {
            "i2v-14B": "/home/admin/resource/model/909204ff.Wan2.1-I2V-14B-720P/v1",
            "flf2v-14B": "/home/admin/resource/model/909204ff.Wan2.1-FLF2V-14B-720P/v1/",
            "vace-14B": "/home/admin/resource/model/909204ff.Wan2.1-I2V-14B-480P/v1/Wan2.1-VACE-14B",
            "vace-1.3B": "../VACE/models/Wan2.1-VACE-1.3B/",
            "t2v-A14B": "/home/admin/resource/model/909204ff.Wan2.1-I2V-14B-480P/v1/Wan2.2-T2V-A14B",
            "i2v-A14B": "/home/admin/resource/model/909204ff.Wan2.1-I2V-14B-480P/v1/Wan2.2-I2V-A14B",
            "lf2v-A14B": "/home/admin/resource/model/909204ff.Wan2.1-I2V-14B-480P/v1/Wan2.2-I2V-A14B",
            "flf2v-A14B": "/home/admin/resource/model/909204ff.Wan2.1-I2V-14B-480P/v1/Wan2.2-I2V-A14B",
            "kf2v-A14B": "/home/admin/resource/model/909204ff.Wan2.1-I2V-14B-480P/v1/Wan2.2-I2V-A14B",
            "i2v-A14B-flash": "/home/admin/resource/model/909204ff.Wan2.1-I2V-14B-480P/v1/Wan2.2-I2V-A14B-Flash",
            "lf2v-A14B-flash": "/home/admin/resource/model/909204ff.Wan2.1-I2V-14B-480P/v1/Wan2.2-I2V-A14B-Flash",
            "flf2v-A14B-flash": "/home/admin/resource/model/909204ff.Wan2.1-I2V-14B-480P/v1/Wan2.2-I2V-A14B-Flash",
            "kf2v-A14B-flash": "/home/admin/resource/model/909204ff.Wan2.1-I2V-14B-480P/v1/Wan2.2-I2V-A14B-Flash",
        }[args.task]
    elif args.base_model == "480P":
        if "i2v" in args.task:
            shift = 3
            ckpt_dir = "/home/admin/resource/model/909204ff.Wan2.1-I2V-14B-480P/v1"
        else:
            raise NotImplementedError("Current only support i2v on 480P base model")
    print_yellow(f"Inference with flow_shift={shift}")

    # Determine the output directory.
    output_dir = args.output_dir
    if args.model:
        output_dir += f"_steps{steps}"
    elif args.high_model or args.low_model:
        tags = []
        if args.high_model:
            tags.append(f"h_steps{high_steps}")
        if args.low_model:
            tags.append(f"l_steps{low_steps}")
        output_dir += "_" + "_".join(tags)
    else:
        tags = []
        if args.high1_model:
            tags.append(f"h1_steps{high1_steps}")
        if args.high2_model:
            tags.append(f"h2_steps{high2_steps}")
        if args.low3_model:
            tags.append(f"l3_steps{low3_steps}")
        if args.low4_model:
            tags.append(f"l4_steps{low4_steps}")
        output_dir += "_" + "_".join(tags)
    eval_size = args.eval_size.replace("*", "x")
    output_dir = f"{output_dir}-base{args.base_model}-size{eval_size}"

    # Run the evaluation.
    cmd = f"""PYTHONPATH=. python -m torch.distributed.run --nproc_per_node={num_gpus} --master_port=29600 \
        {f"trainer/models/{'wan22' if is_wan22 else 'wan21'}/generate_v2v.py"} \
        --task {args.task} \
        --size {args.eval_size} \
        --ckpt_dir {ckpt_dir} \
        {f"--lora_path {ckpt}" if ckpt else ""} \
        {f"--high_noise_lora_path {high_ckpt}" if high_ckpt else ""} \
        {f"--low_noise_lora_path {low_ckpt}" if low_ckpt else ""} \
        {f"--high_noise_1_lora_path {high1_ckpt}" if high1_ckpt else ""} \
        {f"--high_noise_2_lora_path {high2_ckpt}" if high2_ckpt else ""} \
        {f"--low_noise_3_lora_path {low3_ckpt}" if low3_ckpt else ""} \
        {f"--low_noise_4_lora_path {low4_ckpt}" if low4_ckpt else ""} \
        --sample_shift {shift} \
        --data_file {args.data_file} \
        --output_dir {output_dir} \
        {f"--pattern {args.pattern}" if args.pattern else ""} \
        --max_gen_num {args.max_gen_num}"""
    print(f"Running: {cmd}")
    os.system(cmd)
    os.system(f"tar cvf {output_dir}.tar {output_dir}")


if __name__ == "__main__":
    main()
#--dit_fsdp --t5_fsdp --ulysses_size {num_gpus} \