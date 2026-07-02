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
import numpy as np

import torch
import torch.distributed as dist
from PIL import Image
import re
import torch.nn.functional as F

import decord
import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import save_video, str2bool
#from trainer.datasets import prepare_vace_test_conditions

# 忽略警告
warnings.filterwarnings('ignore')
decord.bridge.set_bridge("torch")  # 确保 decord 输出为 torch.Tensor

def get_vace_data(src_path, frame_num, device, train_fps=16):
    decord.bridge.set_bridge("torch")  # Load frames in torch.tensor format.
    src_vr = decord.VideoReader(src_path)
    fps = src_vr.get_avg_fps()
    num_frames = len(src_vr)
    duration = num_frames / fps

    load_fps = train_fps
    frame_interval = 1.0 if abs(fps - load_fps) < 0.1 else fps / load_fps
    frame_indices = np.arange(0, num_frames, frame_interval).astype(int)
    frame_indices = frame_indices[frame_indices < num_frames]
    frame_indices = frame_indices[:frame_num]  # 只取前 frame_num 帧
    #print("src video info", src_vr.get_avg_fps(), len(src_vr))
    src_video = src_vr.get_batch(frame_indices.tolist())
    src_video = src_video.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
    height, width = src_video.shape[2:]
    new_height = 480
    new_width = 832
    if height != new_height or width != new_width:
        src_video = F.interpolate(src_video, size=(new_height, new_width), mode="bicubic", antialias=True)  # torch.uint8
    src_video = src_video.float().div_(127.5).sub_(1.0) # [0, 255] -> [-1, 1]
    src_video = src_video.transpose(0, 1)  # [T, C, H, W] -> [C, T, H, W]
    src_video = src_video.to(device)
    
    return [src_video]

# ======================
# 示例 Prompt 配置
# ======================
EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image": "examples/i2v_input.JPG",
    },
    "ti2v-5B": {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image": "examples/i2v_input.JPG",
    },
}


# ======================
# 参数解析
# ======================
def get_parser():
    parser = argparse.ArgumentParser(description="Generate video from text/image using WanVACE")
    parser.add_argument("--task", type=str, default="vace-1.3B", choices=list(WAN_CONFIGS.keys()), help="Model name to use.")
    parser.add_argument("--size", type=str, default="480p", choices=list(SIZE_CONFIGS.keys()), help="Output resolution (e.g., 480p, 720p).")
    parser.add_argument("--frame_num", type=int, default=None, help="Number of frames to generate (should be 4n+1).")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to checkpoint directory.")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA weights (.pth).")
    #parser.add_argument("--control_path", type=str, default=None, help="Path to controlnet weights (.pth).")
    parser.add_argument("--high_lora_path", type=str, default=None, help="Path to LoRA weights (.pth).")
    parser.add_argument("--low_lora_path", type=str, default=None, help="Path to LoRA weights (.pth).")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test metadata JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated videos.")
    parser.add_argument("--base_seed", type=int, default=-1, help="Random seed. Negative means auto-generate.")
    parser.add_argument("--sample_steps", type=int, default=None, help="Sampling steps (default: 50 for T2V).")
    parser.add_argument("--sample_shift", type=float, default=None, help="Flow matching shift factor.")
    parser.add_argument("--sample_guide_scale", type=float, default=None, help="CFG scale.")
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"], help="Sampling solver.")
    parser.add_argument("--offload_model", type=str2bool, default=None, help="Offload model to CPU to save GPU memory.")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Run T5 on CPU.")
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Use FSDP for T5 (requires distributed).")
    parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Use FSDP for DiT (requires distributed).")
    parser.add_argument("--ulysses_size", type=int, default=1, help="Ulysses parallelism degree.")
    parser.add_argument("--ring_size", type=int, default=1, help="Ring attention parallelism degree.")
    parser.add_argument("--concat", type=str2bool, default=False, help="Whether to save concatenated source and generated video.")
    return parser


# ======================
# 参数验证
# ======================
def validate_args(args):
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # if args.prompt is None:
    #     args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    # if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
    #     args.image = EXAMPLE_PROMPT[args.task]["image"]

    # if args.task == "i2v-A14B":
    #     assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[args.task], (
        f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"
    )
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
        if args.task == "ti2v-5B":
            wan_v2v = wan.WanTV2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                lora_path=args.lora_path,
                device_id=device,
                rank=rank,
                t5_fsdp=args.t5_fsdp,
                dit_fsdp=args.dit_fsdp,
                use_sp=(args.ulysses_size > 1 or args.ring_size > 1),
                t5_cpu=args.t5_cpu,
            )
            wan_v2v.model.eval().requires_grad_(False)
            logging.info("Wan2.2-5B model loaded successfully.")
        else:
            wan_v2v = wan.WanV2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                high_noise_lora_path=args.high_lora_path,
                low_noise_lora_path=args.low_lora_path,
                device_id=device,
                rank=rank,
                t5_fsdp=args.t5_fsdp,
                dit_fsdp=args.dit_fsdp,
                use_sp=(args.ulysses_size > 1 or args.ring_size > 1),
                t5_cpu=args.t5_cpu,
            )
            wan_v2v.high_noise_model.eval().requires_grad_(False)
            wan_v2v.low_noise_model.eval().requires_grad_(False)
            logging.info("Wan2.2-14B model loaded successfully.")
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
        #prompt = d["prompt"]
        src_path = d.get("src_video_path")
        for instruct in ["将这段视频转化为《冰雪奇缘》风格。", "将这段视频转化为《魔女宅急便》风格。", "将这段视频转化为《蓦然回首》风格。", "将这段视频转化为《薇尔莉特永恒花园》风格。"]:
            try:
                if src_path:
                    prompt = instruct + d["prompt"]
                    if rank==0:
                        print(prompt)
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
                else:
                    prompt = instruct + d["prompt"]
                    tag = ""
                    match = re.search(r"《(.*?)》", prompt)
                    if match:
                        tag = match.group(1).replace(" ", "_")

                    src_video = torch.zeros((3, 81, 480, 832)).to(device)
                    src_video = [src_video]

                    output_filename = f"gen_{idx:05d}_{tag}.mp4"
                    output_path = os.path.join(args.output_dir, output_filename)

                logging.info(
                    f"[{idx+1}/{len(data)}] Generating: "
                    f"src={d['src_video_path']},"
                    f"prompt='{prompt[:50]}...', → {output_path}"
                )

                if os.path.exists(output_path):
                    logging.info(f"skip: output already exists: {output_path}")
                    continue

                print("Loading video from ", d['src_video_path'])
                #src_video = decord.VideoReader(d['src_video_path'])
                src_video  = get_vace_data(d['src_video_path'], args.frame_num, device)
                #print("src_video shape:", src_video[0].shape)
                # 生成
                with torch.no_grad():
                    gen_video = wan_v2v.generate(
                        input_prompt=prompt,
                        input_video=src_video,
                        #size=SIZE_CONFIGS[args.size],
                        frame_num=args.frame_num,
                        shift=args.sample_shift,
                        sample_solver=args.sample_solver,
                        sampling_steps=args.sample_steps,
                        guide_scale=args.sample_guide_scale,
                        seed=args.base_seed + idx,  # 每个样本不同 seed
                        offload_model=args.offload_model,
                    )

                # 保存（仅主进程）
                if rank == 0:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    save_video(
                        tensor=gen_video[None],
                        save_file=output_path,
                        fps=16,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    logging.info(f"Saved to {output_path}")
                    if args.concat == True:
                        pad = src_video[0].shape[1] - gen_video.shape[1]
                        if pad != 0:
                            pad_tensor = torch.zeros((gen_video.shape[0], pad, gen_video.shape[2], gen_video.shape[3]), device=gen_video.device)
                            gen_video = torch.cat([gen_video, pad_tensor], dim=1)

                        gen_video = torch.cat([src_video[0], gen_video], dim=3)  # 拼接源视频和生成视频
                        save_video(
                        tensor=gen_video[None],
                        save_file=output_path[:-4] + "_concat.mp4",
                        fps=16,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1),
                    )
                        logging.info(f"Saved to {output_path}")
                
                del gen_video, src_video  # 释放内存
                torch.cuda.empty_cache()

            except Exception as e:
                logging.error(f"Failed to process item {idx}: {e}", exc_info=True)
                continue

            if dist.is_initialized():
                dist.barrier()

    logging.info("All tasks completed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Process interrupted by user.")
    except Exception as e:
        logging.critical(f"Critical error: {e}", exc_info=True)
        sys.exit(1)
