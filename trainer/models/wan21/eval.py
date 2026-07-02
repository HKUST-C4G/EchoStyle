#!/usr/bin/env python
# generate_v2v.py - Fixed & Robust Version

import argparse
import json
import logging
import os
import sys
import random
from datetime import datetime
import warnings

import torch
import torch.distributed as dist
from PIL import Image
import re

import decord
import wan
from wan import WanVace
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import cache_video, str2bool
#from trainer.datasets import prepare_vace_test_conditions

# 忽略警告
warnings.filterwarnings('ignore')
decord.bridge.set_bridge("torch")  # 确保 decord 输出为 torch.Tensor

# ======================
# 示例 Prompt 配置
# ======================
EXAMPLE_PROMPT = {
    "vace-1.3B": {
        "src_ref_images": 'assets/images/girl.png,assets/images/snake.png',
        "prompt": "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。"
    },
    "vace-14B": {
        "src_ref_images": 'assets/images/girl.png,assets/images/snake.png',
        "prompt": "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。"
    }
}


# ======================
# 参数解析
# ======================
def get_parser():
    parser = argparse.ArgumentParser(description="Generate video from text/image using WanVACE")
    parser.add_argument("--task", type=str, default="vace-1.3B", choices=list(WAN_CONFIGS.keys()), help="Model name to use.")
    parser.add_argument("--size", type=str, default="480p", choices=list(SIZE_CONFIGS.keys()), help="Output resolution (e.g., 480p, 720p).")
    parser.add_argument("--frame_num", type=int, default=81, help="Number of frames to generate (should be 4n+1).")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to checkpoint directory.")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA weights (.pth).")
    parser.add_argument("--control_path", type=str, default=None, help="Path to controlnet weights (.pth).")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test metadata JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated videos.")
    parser.add_argument("--base_seed", type=int, default=-1, help="Random seed. Negative means auto-generate.")
    parser.add_argument("--sample_steps", type=int, default=None, help="Sampling steps (default: 50 for T2V).")
    parser.add_argument("--sample_shift", type=float, default=16.0, help="Flow matching shift factor.")
    parser.add_argument("--sample_guide_scale", type=float, default=5.0, help="CFG scale.")
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"], help="Sampling solver.")
    parser.add_argument("--offload_model", type=str2bool, default=None, help="Offload model to CPU to save GPU memory.")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Run T5 on CPU.")
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Use FSDP for T5 (requires distributed).")
    parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Use FSDP for DiT (requires distributed).")
    parser.add_argument("--ulysses_size", type=int, default=1, help="Ulysses parallelism degree.")
    parser.add_argument("--ring_size", type=int, default=1, help="Ring attention parallelism degree.")
    return parser


# ======================
# 参数验证
# ======================
def validate_args(args):
    assert args.ckpt_dir is not None, "Checkpoint directory must be specified."
    assert args.task in WAN_CONFIGS, f"Unsupported model: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"No example prompt for model: {args.task}"
    assert args.size in SUPPORTED_SIZES[args.task], \
        f"Unsupported size {args.size} for {args.task}. Supported: {SUPPORTED_SIZES[args.task]}"

    if args.sample_steps is None:
        args.sample_steps = 50
    if args.base_seed < 0:
        args.base_seed = random.randint(0, sys.maxsize)

    return args


# ======================
# 日志初始化
# ======================
def _init_logging(rank):
    level = logging.INFO if rank == 0 else logging.ERROR
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


# ======================
# 主函数
# ======================
def main():
    parser = get_parser()
    args = parser.parse_args()
    args = validate_args(args)

    # 设置分布式环境变量默认值（用于单卡调试）
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    rank = int(os.getenv("RANK"))
    local_rank = int(os.getenv("LOCAL_RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    device = local_rank

    _init_logging(rank)
    logging.info(f"Process {rank} (local {local_rank}) started.")

    # 自动设置 offload_model
    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model not specified → set to {args.offload_model}")

    # ========== 分布式初始化 ==========
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        logging.info(f"Distributed initialized: world_size={world_size}, rank={rank}")
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), "FSDP requires distributed mode."
        assert not (args.ulysses_size > 1 or args.ring_size > 1), "Parallelism requires distributed mode."

    # ========== xfuser 并行初始化（仅当启用时）==========
    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, \
            f"ulysses_size * ring_size must equal WORLD_SIZE ({world_size})"
        try:
            from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
            init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=args.ring_size,
                ulysses_degree=args.ulysses_size,
            )
            logging.info(f"Initialized Ulysses={args.ulysses_size}, Ring={args.ring_size}")
        except ImportError:
            logging.error("xfuser not installed. Cannot use Ulysses/Ring parallelism.")
            sys.exit(1)

    # ========== 模型配置 ==========
    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, "`num_heads` must be divisible by `ulysses_size`"

    # ========== 广播种子 ==========
    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Args: {args}")
    logging.info(f"Model config: {cfg}")

    # ========== 加载模型 ==========
    try:
        wan_vace = WanVace(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            lora_path=args.lora_path,
            control_path=args.control_path,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )
        wan_vace.model.eval().requires_grad_(False)
        logging.info("WanVace model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)

    # ========== 加载数据 ==========
    if not os.path.exists(args.data_path):
        logging.error(f"Data file not found: {args.data_path}")
        sys.exit(1)

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logging.info(f"Loaded {len(data)} samples from {args.data_path}.")

    # ========== 开始生成 ==========
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, d in enumerate(data):
        try:
            prompt = d["prompt"]
            output_filename_base, extension = os.path.splitext(os.path.basename(d["src_video_path"]))

            tag = ""
            match = re.search(r"《(.*?)》", prompt)
            if match:
                tag = match.group(1).replace(" ", "_")
                
            if tag:
                output_filename = f"{output_filename_base}_{tag}_gen{extension}"
            else:
                output_filename = f"{output_filename_base}_gen{extension}"

            output_path = os.path.join(args.output_dir, output_filename)

            logging.info(
                f"[{idx+1}/{len(data)}] Generating: "
                f"src={d['src_video_path']}, ref={d.get('vace_ref_image_paths', 'N/A')}, "
                f"prompt='{prompt[:50]}...', → {output_path}"
            )

            if os.path.exists(output_path):
                logging.info(f"skip: output already exists: {output_path}")
                continue

            print(d['src_video_path'], d['vace_ref_image_paths'] if 'vace_ref_image_paths' in d else None)
            vace_video, vace_mask, vace_ref_images = wan_vace.prepare_source([d['src_video_path']],
                                                        [None],
                                                        [None],
                                                        args.frame_num, SIZE_CONFIGS[args.size], device)

            # 生成
            with torch.no_grad():
                gen_video = wan_vace.generate(
                    input_prompt=prompt,
                    input_frames=vace_video,
                    input_masks=vace_mask,
                    input_ref_images=vace_ref_images,
                    size=SIZE_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=args.base_seed + idx,  # 每个样本不同 seed
                    offload_model=args.offload_model,
                )
            del vace_video, vace_mask, vace_ref_images  # 释放内存

            # 保存（仅主进程）
            if rank == 0:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cache_video(
                    tensor=gen_video[None],
                    save_file=output_path,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1)
                )
                logging.info(f"Saved to {output_path}")
            del gen_video  # 释放内存
            torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Failed to process item {idx}: {e}", exc_info=True)
            continue

    logging.info("All tasks completed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Process interrupted by user.")
    except Exception as e:
        logging.critical(f"Critical error: {e}", exc_info=True)
        sys.exit(1)
