import argparse
import json
import logging
import os
import re
import sys
import warnings

from trainer.datasets import prepare_vace_test_conditions

warnings.filterwarnings("ignore")

import random

import torch
import torch.distributed as dist
import wan
from PIL import Image
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, str2bool

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image": "examples/i2v_input.JPG",
    },
    "flf2v-14B": {
        "prompt": "CG动画风格，一只蓝色的小鸟从地面起飞，煽动翅膀。小鸟羽毛细腻，胸前有独特的花纹，背景是蓝天白云，阳光明媚。镜跟随小鸟向上移动，展现出小鸟飞翔的姿态和天空的广阔。近景，仰视视角。",
        "first_frame": "examples/flf2v_input_first_frame.png",
        "last_frame": "examples/flf2v_input_last_frame.png",
    },
    "vace-1.3B": {
        "src_ref_images": "examples/girl.png,examples/snake.png",
        "prompt": "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。",
    },
    "vace-14B": {
        "src_ref_images": "examples/girl.png,examples/snake.png",
        "prompt": "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 50
        if "i2v" in args.task:
            args.sample_steps = 40

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0
        elif "flf2v" in args.task or "vace" in args.task:
            args.sample_shift = 16

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[args.task], (
        f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"
    )

def _parse_args():
    parser = argparse.ArgumentParser(description="Generate a image or video from a text prompt or image using Wan")
    parser.add_argument(
        "--task", type=str, default="t2v-14B", choices=list(WAN_CONFIGS.keys()), help="The task to run."
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image.",
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1",
    )
    parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.",
    )
    parser.add_argument("--ulysses_size", type=int, default=1, help="The size of the ulysses parallelism in DiT.")
    parser.add_argument("--ring_size", type=int, default=1, help="The size of the ring attention parallelism in DiT.")
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Whether to use FSDP for T5.")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Whether to place T5 model on CPU.")
    parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file", type=str, default=None, help="The file to save the generated image or video to."
    )
    parser.add_argument("--prompt", type=str, default=None, help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend", action="store_true", default=False, help="Whether to use prompt extend."
    )
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.",
    )
    parser.add_argument("--prompt_extend_model", type=str, default=None, help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.",
    )
    parser.add_argument("--base_seed", type=int, default=-1, help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image", type=str, default=None, help="[image to video] The image to generate the video from."
    )
    parser.add_argument(
        "--first_frame",
        type=str,
        default=None,
        help="[first-last frame to video] The image (first frame) to generate the video from.",
    )
    parser.add_argument(
        "--last_frame",
        type=str,
        default=None,
        help="[first-last frame to video] The image (last frame) to generate the video from.",
    )
    parser.add_argument(
        "--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"], help="The solver used to sample."
    )
    parser.add_argument("--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift", type=float, default=None, help="Sampling shift factor for flow matching schedulers."
    )
    parser.add_argument("--sample_guide_scale", type=float, default=5.0, help="Classifier free guidance scale.")
    ############################
    # Modified by wan_trainer. #
    ############################
    parser.add_argument("--lora_path", type=str, default=None, help="The path to the lora file.")
    parser.add_argument("--data_file", type=str, default=None, help="JSON file for inference.")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory.")
    parser.add_argument("--pattern", type=str, default=None, help="Pattern to filter data.")
    parser.add_argument("--max_gen_num", type=int, default=-1, help="Max generation number.")

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
            #################################################################################################
            # Modified by wan_trainer to override the logging handlers added by some third-party libraries. #
            #################################################################################################
            force=True,
        )
    else:
        logging.basicConfig(level=logging.ERROR)

def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), (
            f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        )
        assert not (args.ulysses_size > 1 or args.ring_size > 1), (
            f"context parallel are not supported in non-distributed environments."
        )

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, (
            f"The number of ulysses_size and ring_size should be equal to the world size."
        )
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    # ========================
    # 加载 Prompt 列表
    # ========================
    if args.data_file is not None:
        # Rank 0 读取 JSON
        if rank == 0:
            try:
                with open(args.data_file, 'r', encoding='utf-8') as f:
                    data_list = json.load(f)
                # 验证格式
                for item in data_list:
                    assert "id" in item and "prompt" in item, f"Missing 'id' or 'prompt' in item: {item}"
                logging.info(f"Loaded {len(data_list)} prompts from {args.data_file}")
            except Exception as e:
                logging.error(f"Failed to load data file: {e}")
                sys.exit(1)
        else:
            data_list = None

        # 广播给所有 rank
        if dist.is_initialized():
            object_list = [data_list]
            dist.broadcast_object_list(object_list, src=0)
            data_list = object_list[0]
    else:
        # 单 prompt 模式（原有逻辑）
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        data_list = [{"id": "single", "prompt": args.prompt}]

    # ========================
    # 初始化 Prompt Expander（如果需要）
    # ========================
    prompt_expander = None
    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model, is_vl="i2v" in args.task or "flf2v" in args.task
            )
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model, is_vl="i2v" in args.task, device=rank
            )
        else:
            raise NotImplementedError(f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, (
            f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."
        )

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    # ========================
    # 初始化模型（只初始化一次）
    # ========================
    if "t2v" in args.task or "t2i" in args.task:
        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        # ========================
        # 创建输出目录
        # ========================
        if rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            logging.info(f"Output directory: {args.output_dir}")

        # ========================
        # 遍历每条 prompt 生成
        # ========================
        for idx, item in enumerate(data_list):
            original_id = item["id"]
            prompt_text = item["prompt"]

            # 扩展 prompt（如果启用）
            if args.use_prompt_extend:
                if rank == 0:
                    logging.info(f"[{original_id}] Extending prompt ...")
                    prompt_output = prompt_expander(
                        prompt_text, tar_lang=args.prompt_extend_target_lang, seed=args.base_seed + idx
                    )
                    if prompt_output.status == False:
                        logging.warning(f"[{original_id}] Prompt extension failed: {prompt_output.message}. Using original.")
                        final_prompt = prompt_text
                    else:
                        final_prompt = prompt_output.prompt
                    final_prompt_list = [final_prompt]
                else:
                    final_prompt_list = [None]

                if dist.is_initialized():
                    dist.broadcast_object_list(final_prompt_list, src=0)
                final_prompt = final_prompt_list[0]
            else:
                final_prompt = prompt_text

            logging.info(f"[Rank {rank}] Generating for ID: {original_id},prompt: {final_prompt}")

            # 生成
            video = wan_t2v.generate(
                final_prompt,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed + idx,  # 每个样本不同 seed
                offload_model=args.offload_model,
            )

            # 保存（仅 rank 0）
            if rank == 0:
                # 格式化 id 为 001, 002 等（如果 id 是数字）
                if isinstance(original_id, int):
                    filename = f"{original_id:03d}.mp4"
                else:
                    # 清理非法字符
                    safe_id = re.sub(r'[^\w\-_\.]', '_', str(original_id))
                    filename = f"{safe_id}.mp4"

                save_path = os.path.join(args.output_dir, filename)
                cache_video(
                    tensor=video[None],
                    save_file=save_path,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1)
                )
                logging.info(f"Saved video for ID {original_id} to {save_path}")

                # 可选：保存 prompt 文本
                prompt_txt_path = os.path.join(args.output_dir, filename.replace(".mp4", "_prompt.txt"))
                with open(prompt_txt_path, "w", encoding="utf-8") as f:
                    f.write(final_prompt)

    else:
        raise NotImplementedError(f"Task {args.task} not supported in batch mode yet.")

    logging.info("All generation tasks completed.")

if __name__ == "__main__":
    args = _parse_args()
    generate(args)

