import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from torchvision.transforms.functional import to_tensor
import torchvision

#from ..utils import resize_and_center_crop
from .base_dataset import BaseDataset
#from base_dataset import BaseDataset
logger = logging.getLogger(__name__)


def _get_vace_mask(video_shape) -> torch.Tensor:
    #生成mask，根据mode进行掩码处理
    vace_mask = torch.ones((1, video_shape[1], video_shape[2], video_shape[3]))
    return vace_mask


def _get_vace_ref_images(vace_ref_image_paths, image_size) -> List[torch.Tensor] | None:
    #ref_images预处理，进行resize
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
    #vace_modes: List[str] = None,
    #vace_ref_image_paths: List[str] = None,
    video: torch.Tensor = None,
    src_video: torch.Tensor = None,
    **kwargs,
):
    vace_video = src_video.clone()
    vace_mask = _get_vace_mask( video.shape)
    #vace_ref_images = _get_vace_ref_images(vace_ref_image_paths, video.shape[-2:])
    return vace_video, vace_mask


def prepare_vace_test_conditions(
    #vace_modes: List[str] = None,
    #vace_key_frame_index: int = None,
    vace_ref_image_paths: str = None,
    video: torch.Tensor = None,
    src_video: torch.Tensor = None,
    # first_frame_path: str = None,
    # key_frame_path: str = None,
    # last_frame_path: str = None,
    #max_pixels: int = None,
    **kwargs,
):
    # Calculate the shape of the condition video.

    # [3, T, H, W], range [-1, 1], fp32
    vace_video = src_video.clone()
    vace_mask = _get_vace_mask(src_video.shape)
    vace_ref_images = _get_vace_ref_images(vace_ref_image_paths, src_video.shape[-2:])
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

def save_tensor_as_video(video_tensor, output_path, fps=16): # 使用较低的fps以便观察
    # ... (代码同上) ...
    # 0. 确保张量在 CPU 上，并且是 float 类型
    video_tensor = video_tensor.cpu().float()
    # 1. 反归一化
    if video_tensor.min() >= 0 and video_tensor.max() <= 1:
        video_tensor = video_tensor * 255
    else:
        video_tensor = (video_tensor + 1) / 2 * 255
    # 2. 转换数据类型
    video_tensor = video_tensor.to(torch.uint8)
    # 3. 调整维度顺序
    video_tensor = video_tensor.permute(1, 2, 3, 0)
    # 4. 保存视频
    try:
        torchvision.io.write_video(output_path, video_tensor, fps=fps, video_codec='h264')
        print(f"✅ 视频成功保存至: {output_path}")
    except Exception as e:
        print(f"❌ 保存视频失败: {e}")

class Wanexv2vDataset(BaseDataset):
    def get_data(self, idx):
        """
        The function will be called by BaseDataset.__getitem__()
        """
        data = super().get_vace_data(idx)
        #print("video-shape:", data["video"].shape, data["src_video"].shape)
        #data.update(dict(vace_video=vace_video, vace_mask=vace_mask, vace_ref_images=[]))
        return data

    def cache_fn(self, idx, vae, text_encoder, cache_path, *args):
        """
        The function will be called by BaseDataset.calculate_or_load_dataset_cache()
        """
        data = self[idx]
        video = data["video"].cuda()  # [3, T, H, W], fp32
        src_video = data["src_video"].cuda()

        h, w = video.shape[-2:]
        start_col = (w - h) // 2
        end_col = start_col + h
        video = video[:, :, :, start_col:end_col]
        src_video = src_video[:, :, :, start_col:end_col]

        print("video-shape:", video.shape, " src_video-shape:", src_video.shape)
        assert video.shape[1] == src_video.shape[1], "video and src_video should have the same frame number"
        src_video[:, :16, :, :] = video[:, :16, :, :]  # make sure the first frame is the same

        prompt = data["prompt"]
        context = text_encoder([prompt], "cuda")[0]  # [n, 4096]?

        #print("context-shape:", context.shape)
        #video_latent = _vace_encode_frames([video], None, None, vae)[0]
        video_latent = vae.encode([video])[0]  # [1, 16, 138, 102]
        src_video_latent = vae.encode([src_video])[0]
        # print(" video_latent-shape:", video_latent.shape, " src_video_latent-shape:", src_video_latent.shape)

        save_data = {"context": context, "video_latent": video_latent, "src_context": src_video_latent}
        filename = os.path.basename(data["video_path"])
        video_path = os.path.join(cache_path, filename)
        save_file(save_data, os.path.splitext(video_path)[0] + ".safetensors")

# if __name__ == "__main__":
    
#     # 用来保存调试输出视频的目录
#     debug_output_dir = "./debug_videos"
#     os.makedirs(debug_output_dir, exist_ok=True)
    
#     # 要检查的数据集样本数量
#     num_samples_to_check = 3
#     try:
#         dataset = Wanv2vDataset(
#         data_path="../datasets/cartoon20k/traindata_480p_src/metadata_debug.json",
#         frame_bucket=[17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61],
#         train_fps=24,
#         max_pixels=901120)
#         print(f"数据集加载成功，总共有 {len(dataset)} 个样本。")
#     except Exception as e:
#         print(f"❌ 实例化数据集失败: {e}")
#         print("    请确保 `dataset_config` 包含了所有必要的参数。")

#     # --- 3. 循环遍历并检查样本 ---
#     print(f"\n--- 开始检查前 {num_samples_to_check} 个样本 ---")
#     for i in range(min(num_samples_to_check, len(dataset))):
#         print(f"\n--- 正在处理样本索引: {i} ---")
        
#         # 从数据集中获取一个样本
#         # __getitem__ 会被调用，然后是 get_data
#         try:
#             data = dataset[i]
#         except Exception as e:
#             print(f"❌ 获取样本 {i} 时出错: {e}")
#             continue

#         # 提取 'video' 和 'src_video'
#         video_tensor = data.get("video")
#         src_video_tensor = data.get("src_video")

#         if video_tensor is None or src_video_tensor is None:
#             print("❌ 样本数据中缺少 'video' 或 'src_video'。")
#             continue

#         # --- 4. 打印信息并保存视频 ---
#         print("--- 'video' (目标视频) ---")
#         print(f"  Shape: {video_tensor.shape}")
#         print(f"  Dtype: {video_tensor.dtype}")
#         print(f"  Min value: {video_tensor.min():.4f}, Max value: {video_tensor.max():.4f}")
        
#         # 定义保存路径
#         video_output_path = os.path.join(debug_output_dir, f"sample_{i}_video.mp4")
#         save_tensor_as_video(video_tensor, video_output_path)
        
#         print("\n--- 'src_video' (源视频) ---")
#         print(f"  Shape: {src_video_tensor.shape}")
#         print(f"  Dtype: {src_video_tensor.dtype}")
#         print(f"  Min value: {src_video_tensor.min():.4f}, Max value: {src_video_tensor.max():.4f}")
        
#         src_video_output_path = os.path.join(debug_output_dir, f"sample_{i}_src_video.mp4")
#         save_tensor_as_video(src_video_tensor, src_video_output_path)
