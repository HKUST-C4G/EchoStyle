import logging
import os
from typing import List

import numpy as np
import decord
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from torchvision.transforms.functional import to_tensor

from ..utils import resize_and_center_crop
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def _load_image(image_path, video_shape) -> torch.Tensor:
    """
    Load an image and resize it to the shape of the video.
    """
    image = None
    if image_path is not None:
        image = Image.open(image_path).convert("RGB")
        image = to_tensor(image).mul_(2).sub_(1)  # shape [3, H, W], range [-1, 1]
        image = resize_and_center_crop(image, video_shape[-2:])
    return image


def _get_vace_video(   
        #根据video和mode对vace_video进行插帧
    vace_modes,
    vace_key_frame_index=-1,
    # Train
    video=None,
    # Test
    video_shape=None,
    first_frame_path=None,
    key_frame_path=None,
    last_frame_path=None,
) -> torch.Tensor:
    first_frame = _load_image(first_frame_path, video_shape)
    key_frame = _load_image(key_frame_path, video_shape)
    last_frame = _load_image(last_frame_path, video_shape)

    # 0 is equivalent to 127.5 in [0, 255] since video has been normalized to [-1, 1].
    vace_video = torch.zeros_like(video) if video is not None else torch.zeros(video_shape)

    if "i2v" in vace_modes:
        # Keep the first frame as it is, and set other frames to 0.
        vace_video[:, 0, :, :] = video[:, 0, :, :] if video is not None else first_frame
    elif "flf2v" in vace_modes:
        # Keep the first and last frame as they are, and set other frames to 0.
        vace_video[:, 0, :, :] = video[:, 0, :, :] if video is not None else first_frame
        vace_video[:, -1, :, :] = video[:, -1, :, :] if video is not None else last_frame
    elif "lf2v" in vace_modes:
        # Keep the last frame as it is, and set other frames to 0.
        vace_video[:, -1, :, :] = video[:, -1, :, :] if video is not None else last_frame
    elif "kf2v" in vace_modes:
        # Keep the first frame and the key frame as they are, and set other frames to 0.
        vace_video[:, 0, :, :] = video[:, 0, :, :] if video is not None else first_frame
        vace_video[:, vace_key_frame_index, :, :] = (
            video[:, vace_key_frame_index, :, :] if video is not None else key_frame
        )

    return vace_video


def _get_vace_mask(vace_modes, vace_key_frame_index, video_shape) -> torch.Tensor:
    #生成mask，根据mode进行掩码处理
    vace_mask = torch.ones((1, video_shape[1], video_shape[2], video_shape[3]))
    # if "i2v" in vace_modes:
    #     vace_mask[:, 0, :, :] = 0
    # elif "flf2v" in vace_modes:
    #     vace_mask[:, 0, :, :] = 0
    #     vace_mask[:, -1, :, :] = 0
    # elif "lf2v" in vace_modes:
    #     vace_mask[:, -1, :, :] = 0
    # elif "kf2v" in vace_modes:
    #     vace_mask[:, 0, :, :] = 0
    #     vace_mask[:, vace_key_frame_index, :, :] = 0
    return vace_mask


def _get_vace_ref_images(vace_modes, vace_ref_image_paths, image_size) -> List[torch.Tensor] | None:
    #ref_images预处理，进行resize
    if "ref_images" not in vace_modes:
        return None
    vace_ref_images = []
    for ref_img in vace_ref_image_paths:
        ref_img = Image.open(ref_img).convert("RGB")
        ref_img = to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
        if ref_img.shape[-2:] != image_size:
            canvas_height, canvas_width = image_size
            ref_height, ref_width = ref_img.shape[-2:]
            white_canvas = torch.ones((3, 1, canvas_height, canvas_width))  # [-1, 1]
            scale = min(canvas_height / ref_height, canvas_width / ref_width)
            new_height = int(ref_height * scale)
            new_width = int(ref_width * scale)
            ref_img = ref_img.squeeze(1).unsqueeze(0)
            resized_image = F.interpolate(ref_img, size=(new_height, new_width), mode="bilinear", align_corners=False)
            resized_image = resized_image.squeeze(0).unsqueeze(1)
            top = (canvas_height - new_height) // 2
            left = (canvas_width - new_width) // 2
            white_canvas[:, :, top : top + new_height, left : left + new_width] = resized_image
            ref_img = white_canvas
        vace_ref_images.append(ref_img)
    return vace_ref_images


def _prepare_vace_train_conditions(
    vace_modes: List[str] = None,
    #vace_key_frame_index: int = None,
    vace_ref_image_paths: List[str] = None,
    src_video: torch.Tensor = None,
    **kwargs,
):
    vace_video = src_video.clone()
    vace_mask = _get_vace_mask(vace_modes, src_video.shape)
    vace_ref_images = _get_vace_ref_images(vace_modes, vace_ref_image_paths, src_video.shape[-2:])
    return vace_video, vace_mask, vace_ref_images


def prepare_vace_test_conditions(
    vace_modes: List[str] = None,
    vace_key_frame_index: int = None,
    vace_ref_image_paths: str = None,
    first_frame_path: str = None,
    key_frame_path: str = None,
    last_frame_path: str = None,
    max_pixels: int = None,
    **kwargs,
):
    # Calculate the shape of the condition video.
    frame_path = first_frame_path or key_frame_path or last_frame_path or vace_ref_image_paths[0]
    if frame_path:
        width, height = Image.open(frame_path).size
        scale_factor = np.sqrt(max_pixels / (height * width))
        new_height, new_width = round(height * scale_factor), round(width * scale_factor)
        spatial_downsample_factor = 16  # 8 (vae) * 2 (patchfy)
        new_height = new_height // spatial_downsample_factor * spatial_downsample_factor
        new_width = new_width // spatial_downsample_factor * spatial_downsample_factor
        video_shape = (3, 81, new_height, new_width)

    # [3, T, H, W], range [-1, 1], fp32
    vace_video = _get_vace_video(
        vace_modes=vace_modes,
        vace_key_frame_index=vace_key_frame_index,
        video_shape=video_shape,
        first_frame_path=first_frame_path,
        key_frame_path=key_frame_path,
        last_frame_path=last_frame_path,
    )
    # [1, T, H, W], value in {0, 1}, fp32
    vace_mask = _get_vace_mask(vace_modes, vace_key_frame_index, video_shape)
    # List of [3, 1, H, W], range [-1, 1], fp32
    vace_ref_images = _get_vace_ref_images(vace_modes, vace_ref_image_paths, video_shape[-2:])
    return vace_video, vace_mask, vace_ref_images


def _vace_encode_frames(frames, ref_images, masks, vae):
    if ref_images is None:
        ref_images = [None] * len(frames)
    else:
        assert len(frames) == len(ref_images)

    if masks is None:
        latents = vae.encode(frames)
    else:
        masks = [torch.where(m > 0.5, 1.0, 0.0) for m in masks]
        inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
        reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
        inactive = vae.encode(inactive)
        reactive = vae.encode(reactive)
        # [tensor([32, 21, 138, 102]),]
        latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]

    cat_latents = []
    for latent, refs in zip(latents, ref_images):
        if refs is not None:
            if masks is None:
                ref_latent = vae.encode(refs)
            else:
                # [tensor([16, 1, 138, 102]),]
                ref_latent = vae.encode(refs)
                # [tensor([32, 1, 138, 102]),]
                ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
            assert all([x.shape[1] == 1 for x in ref_latent])
            # [32, 22, 138, 102]
            latent = torch.cat([*ref_latent, latent], dim=1)
        cat_latents.append(latent)
    return cat_latents


def _vace_encode_masks(masks, ref_images=None, vae_stride=(4, 8, 8)):
    if ref_images is None:
        ref_images = [None] * len(masks)
    else:
        assert len(masks) == len(ref_images)

    result_masks = []
    for mask, refs in zip(masks, ref_images):
        _, depth, height, width = mask.shape
        new_depth = int((depth + 3) // vae_stride[0])
        height = 2 * (int(height) // (vae_stride[1] * 2))
        width = 2 * (int(width) // (vae_stride[2] * 2))

        # Reshape
        mask = mask[0, :, :, :]
        mask = mask.view(depth, height, vae_stride[1], width, vae_stride[1])  # depth, height, 8, width, 8
        mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
        mask = mask.reshape(vae_stride[1] * vae_stride[2], depth, height, width)  # 8*8, depth, height, width

        # Interpolation
        # [64, 21, 138, 102]
        mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode="nearest-exact").squeeze(0)

        if refs is not None:
            length = len(refs)
            mask_pad = torch.zeros_like(mask[:, :length, :, :])
            # [64, 22, 138, 102]
            mask = torch.cat((mask_pad, mask), dim=1)
        result_masks.append(mask)
    return result_masks

class VACEDataset(BaseDataset):
    def get_data(self, idx):
        """
        The function will be called by BaseDataset.__getitem__()
        """
        data = super().get_vace_data(idx)

        vace_video, vace_mask, vace_ref_images = _prepare_vace_train_conditions(
            **self.data_infos[idx], src_video=data["src_video"]
        )

        data.update(dict(vace_video=vace_video, vace_mask=vace_mask, vace_ref_images=vace_ref_images))
        return data

    def cache_fn(self, idx, vae, text_encoder, *args):
        """
        The function will be called by BaseDataset.calculate_or_load_dataset_cache()
        """
        data = self[idx]
        video = data["video"].cuda()  # [3, T, H, W], fp32
        vace_video = data["vace_video"].cuda()
        vace_mask = data["vace_mask"].cuda()
        vace_ref_images = [x.cuda() for x in data["vace_ref_images"]] if data["vace_ref_images"] else None
        prompt = data["prompt"]

        context = text_encoder([prompt], "cuda")[0]  # [70, 4096]

        video_latent = _vace_encode_frames([video], [vace_ref_images], None, vae)[0]  # [16, 22, 138, 102] 

        z0 = _vace_encode_frames([vace_video], [vace_ref_images], [vace_mask], vae) #video的context
        m0 = _vace_encode_masks([vace_mask], [vace_ref_images])
        z = torch.cat([z0[0], m0[0]], dim=0)  # [96, 22, 138, 102]

        save_data = {"context": context, "video_latent": video_latent, "vace_context": z}
        save_file(save_data, os.path.splitext(data["video_path"])[0] + ".safetensors")
