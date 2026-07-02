import logging
import os
import random
from typing import Optional

import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from .distributed import get_rank

logger = logging.getLogger(__name__)


def set_random_seed(seed: Optional[int] = None, deterministic: bool = False) -> None:
    """
    Args:
        seed (int): If None or negative, use a generated seed.
        deterministic (bool): If True, set the deterministic option for CUDNN backend.
    """
    if seed is None or seed < 0:
        new_seed = np.random.randint(2**32)
        logger.info(f"Got invalid seed: {seed}, will use the randomly generated seed: {new_seed}")
        seed = new_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Set random seed to {seed}.")
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logger.info(
            "The CUDNN is set to deterministic. This will increase reproducibility, "
            "but may slow down your training considerably."
        )


def resize_and_center_crop(image, target_size):
    """
    Args:
        image (torch.Tensor): [3, H, W]
        target_size (tuple): (H, W)

    Returns:
        torch.Tensor: [3, H, W]
    """
    h, w = image.shape[-2:]
    target_h, target_w = target_size
    if h != target_h or w != target_w:
        scale_factor = max(target_w / w, target_h / h)
        new_size = [round(h * scale_factor), round(w * scale_factor)]
        transform = transforms.Compose([transforms.Resize(new_size), transforms.CenterCrop(target_size)])
        image = transform(image)
    return image


def move_models_to_device(*models, device="cuda"):
    for model in models:
        if hasattr(model, "to"):
            model.to(device)
        if hasattr(model, "model") and hasattr(model.model, "to"):
            model.model.to(device)
    if device == "cpu":
        torch.cuda.empty_cache()


def pad_videos_to_same_size(video_list, padding_val=-1):
    """
    Pad all videos in the list to the same size.

    Args:
        video_list (list): List of video tensors, each with shape [3, T, H, W].
        padding_val (float): Value used for spatial dimension padding, default is -1.

    Returns:
        torch.Tensor: Padded video tensor with shape [batch_size, 3, max_T, max_H, max_W]
    """
    # Calculate maximum dimensions across all videos.
    max_T = max(video.shape[1] for video in video_list)
    max_H = max(video.shape[2] for video in video_list)
    max_W = max(video.shape[3] for video in video_list)

    padded_videos = []
    for video in video_list:
        C, T, H, W = video.shape
        assert C == 3, f"Video must have 3 channels, got {C}"

        # Temporal padding: repeat the last frame.
        if T < max_T:
            last_frame = video[:, -1:, :, :]  # [3, 1, H, W]
            time_padding = last_frame.repeat(1, max_T - T, 1, 1)  # [3, max_T-T, H, W]
            video = torch.cat([video, time_padding], dim=1)  # [3, max_T, H, W]

        # Spatial padding: create tensor filled with padding_val, then place original video in center.
        padded_video = torch.full((C, max_T, max_H, max_W), padding_val, dtype=video.dtype, device=video.device)

        # Calculate position to center the original video in the target tensor.
        h_start = (max_H - H) // 2
        w_start = (max_W - W) // 2
        h_end = h_start + H
        w_end = w_start + W

        # Place original video in the center.
        padded_video[:, :, h_start:h_end, w_start:w_end] = video

        padded_videos.append(padded_video)

    # Stack all videos into a batch.
    return torch.stack(padded_videos, dim=0)  # [batch_size, 3, max_T, max_H, max_W]


class Reporter:
    def __init__(self, args):
        self.rank = get_rank()
        self.report_to = args.report_to
        logging_dir = os.path.join(args.output_dir, args.logging_dir)
        if self.rank == 0 and self.report_to == "wandb":
            logger.info("Connecting to Wandb...")
            exp_name = args.output_dir.replace("training_outputs/wantrainer_", "")
            wandb.init(project="wan_lora", name=exp_name, config=args, dir=logging_dir)
        elif self.rank == 0 and self.report_to == "tensorboard":
            logger.info("Init TensorBoard...")
            self.tb_writer = SummaryWriter(logging_dir)

    def log_loss(self, loss, lr, step):
        if self.rank:
            return
        if self.report_to == "wandb":
            wandb.log({"train/loss": loss, "train/lr": lr}, step)
        elif self.report_to == "tensorboard":
            self.tb_writer.add_scalars("train", {"loss": loss, "lr": lr}, step)

    def log_video(self, videos, step, fps=16, tag="validation/model_pred"):
        """
        Args:
            videos (torch.Tensor): [B, T, C, H, W], in range [0, 255], dtype=torch.uint8
        """
        if self.rank:
            return
        if self.report_to == "wandb":
            wandb.log({tag: wandb.Video(videos.numpy(), format="mp4", fps=fps)}, step)
        elif self.report_to == "tensorboard":
            self.tb_writer.add_video(tag, videos, step, fps=fps)
