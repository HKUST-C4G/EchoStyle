import json
import logging
import os
import random
from PIL import Image

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

# decord must be imported after torch to avoid segmentation fault.
# Ref: https://github.com/dmlc/decord/issues/329
import decord  # isort: skip
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from tqdm import tqdm
import sys

import trainer.utils.distributed as dist_ops

logger = logging.getLogger(__name__)


# def _get_vace_mask(video_shape) -> torch.Tensor:
#     #生成mask，根据mode进行掩码处理
#     vace_mask = torch.ones((1, video_shape[1], video_shape[2], video_shape[3]))
#     return vace_mask


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

class imagev2vDataset(Dataset):
    def __init__(
        self,
        # Path to a JSON file.
        data_path="",
        # Multiple possible video lengths. Video will be assigned to the maximum length in the frame_bucket.
        frame_bucket=[81],
        # Regardless of the fps of the raw data, always train at a fixed fps.
        train_fps=16,
        # If provided, train at a fixed duration by speeding up the video.
        train_duration=None,
        # Maximum height * width for video resolution.
        max_pixels=720 * 1280,
        # VAE: [4, 8, 8], DiT patch size: [1, 2, 2]
        spatial_downsample_factor=16,
    ):
        super().__init__()

        self.data_path = data_path
        self.frame_bucket = frame_bucket
        self.train_fps = train_fps
        self.train_duration = train_duration
        self.max_pixels = max_pixels
        self.spatial_downsample_factor = spatial_downsample_factor

        self.data_infos = self.preprocess(self.data_path)

    def __getitem__(self, idx):
        try:
            if hasattr(self, "cached_data"):
                return self.cached_data[idx]
            else:
                return self.get_data(idx)
        except Exception as e:
            logger.error(f"Data loading error: {e}\nRetry with another data.")
            return self.__getitem__(random.randint(0, len(self) - 1))

    def __len__(self):
        return len(self.data_infos)
    
    def get_vace_data_from_images(self, idx):
        """
        加载一对成对的图像，并将每张图像复制成 target_length 帧的“视频”，
        然后进行后续处理。
        """
        info = self.data_infos[idx]

        # 1. 使用 Pillow 加载图像
        try:
            image = Image.open(info["video_path"]).convert("RGB")
            src_image = Image.open(info["src_video_path"]).convert("RGB")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"无法找到图像文件: {e}. 请确保路径正确。")
        
        # 2. 将 PIL.Image 转换为 PyTorch 张量，格式为 (C, H, W)
        # PIL (H, W, C) -> NumPy (H, W, C) -> PyTorch (H, W, C)
        # -> permute to (C, H, W)
        single_frame = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        src_single_frame = torch.from_numpy(np.array(src_image)).permute(2, 0, 1)

        # 3. 将单帧图像复制成多帧“视频”
        # 使用 unsqueeze(0) 增加一个时间维度，变为 (1, C, H, W)
        # 然后使用 repeat 在时间维度上复制 target_length 次
        # 最终得到 (target_length, C, H, W) 形状的张量
        video = single_frame.unsqueeze(0).repeat(41, 1, 1, 1)
        src_video = src_single_frame.unsqueeze(0).repeat(41, 1, 1, 1)
        
        # 此时 video 和 src_video 的形状是 [81, C, H, W]，类型是 torch.uint8
        # 这个形状与原代码中 [T, C, H, W] 的格式完全匹配

        # --- 后续的尺寸调整和归一化逻辑保持不变 ---

        height, width = video.shape[2:]
        # scale_factor = np.sqrt(self.max_pixels / (height * width))
        # new_height, new_width = round(height * scale_factor), round(width * scale_factor)
        new_height = height // self.spatial_downsample_factor * self.spatial_downsample_factor
        new_width = width // self.spatial_downsample_factor * self.spatial_downsample_factor

        if height != new_height or width != new_width:
            video = F.interpolate(video, size=(new_height, new_width), mode="bicubic", antialias=True)
            src_video = F.interpolate(src_video, size=(new_height, new_width), mode="bicubic", antialias=True)

        video = video.float().div_(127.5).sub_(1.0)  # [0, 255] -> [-1, 1]
        src_video = src_video.float().div_(127.5).sub_(1.0)  # [0, 255] -> [-1, 1]
        
        # 将 [T, C, H, W] -> [C, T, H, W]，其中 T=81
        video = video.transpose(0, 1)
        src_video = src_video.transpose(0, 1)
        
        assert src_video.shape == video.shape, \
            f"Shape mismatch: src_video {src_video.shape} vs video {video.shape}"
            
        # 您注释掉的 padding 逻辑在这里不再需要，因为我们已经生成了固定长度的视频
        
        return dict(video=video, src_video=src_video, 
                    video_path=info["video_path"], 
                    src_video_path=info["src_video_path"],
                    prompt=info["prompt"])
    
    def preprocess(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            data_infos = json.load(f)
        return data_infos

    def get_data(self, idx):
        """
        The function will be called by BaseDataset.__getitem__()
        """
        data = self.get_vace_data_from_images(idx)
        print("video-shape:", data["video"].shape, data["src_video"].shape)
        return data

    def cache_fn(self, idx, vae, text_encoder, cache_path, *args):
        """
        The function will be called by BaseDataset.calculate_or_load_dataset_cache()
        """
        data = self[idx]
        video = data["video"].cuda()  # [3, T, H, W], fp32
        vace_video = data["src_video"].cuda()
        vace_mask = torch.ones_like(video[0:1, :, :, :]).cuda()
        #vace_ref_images = [x.cuda() for x in data["vace_ref_images"]] if data["vace_ref_images"] else None
        prompt = data["prompt"]

        context = text_encoder([prompt], "cuda")[0]  # [70, 4096]?
        print(data["src_video_path"])

        #print("context-shape:", context.shape)

        video_latent = _vace_encode_frames([video], None, None, vae)[0]
        #print(video_latent.shape)

        z0 = _vace_encode_frames([vace_video], None, [vace_mask], vae)[0] #video的context
        #print(z0.shape)
        m0 = _vace_encode_masks([vace_mask])[0]
        z = torch.cat([z0, m0], dim=0)  # [96, 22, 138, 102]
        #print("z-shape:", z.shape)

        save_data = {"context": context, "video_latent": video_latent, "vace_context": z}
        filename = os.path.basename(data["video_path"])
        video_path = os.path.join(cache_path, filename)
        save_file(save_data, os.path.splitext(video_path)[0] + ".safetensors")

    def calculate_or_load_dataset_cache(self, vae, text_encoder, clip, task, cache_path):
        """
        NOTE: Currently, this function only supports single-node or multi-node with shared file system.
        """
        os.makedirs(cache_path, exist_ok=True)
        rank, world_size = dist_ops.get_rank(), dist_ops.get_world_size()

        video_paths = [info["video_path"] for info in self.data_infos]
        cache_files = []
        for video_path in video_paths:
            filename = os.path.basename(video_path)
            video_path = os.path.join(cache_path, filename)
            cache_file = os.path.splitext(video_path)[0] + ".safetensors"
            if os.path.exists(cache_file):
                cache_files.append(cache_file)

        if len(cache_files) == len(self):
            logger.info(f"Found {len(cache_files)} cached files, loading...")
            cached_data = [load_file(cache_file) for cache_file in cache_files]
            self.cached_data = cached_data
            return

        if len(cache_files) > 0:
            logger.warning(f"Found {len(cache_files)} corrupted cache files, delete them and calculate again...")
            if rank == 0:
                [os.remove(f) for f in cache_files]
        else:
            logger.info("No cached files found, calculating dataset cache...")

        # Calculate the dataset cache distributedly.
        indices_per_rank = list(range(len(self)))[rank::world_size]
        for idx in tqdm(indices_per_rank, desc="Calculating dataset cache"):
            self.cache_fn(idx, vae, text_encoder, cache_path, clip, task)

        # Each process has processed its own data and now needs to load all the data.
        dist.barrier()
        self.calculate_or_load_dataset_cache(vae, text_encoder, clip, task, cache_path)

# if __name__ == "__main__":
    
#     # 用来保存调试输出视频的目录
#     debug_output_dir = "./debug_videos"
#     os.makedirs(debug_output_dir, exist_ok=True)
    
#     # 要检查的数据集样本数量
#     num_samples_to_check = 50
#     try:
#         dataset = VACEv2vDataset(
#         data_path="../datasets/cartoon20k/traindata_480p_src/metadata.json",
#         frame_bucket=[17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81],
#         train_fps=16,)
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
#         #save_tensor_as_video(video_tensor, video_output_path)
        
#         print("\n--- 'src_video' (源视频) ---")
#         print(f"  Shape: {src_video_tensor.shape}")
#         print(f"  Dtype: {src_video_tensor.dtype}")
#         print(f"  Min value: {src_video_tensor.min():.4f}, Max value: {src_video_tensor.max():.4f}")
        
#         src_video_output_path = os.path.join(debug_output_dir, f"sample_{i}_src_video.mp4")
#         #save_tensor_as_video(src_video_tensor, src_video_output_path)
