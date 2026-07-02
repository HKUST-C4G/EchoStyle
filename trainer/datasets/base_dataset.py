import json
import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as F_tv 
from PIL import Image

# decord must be imported after torch to avoid segmentation fault.
# Ref: https://github.com/dmlc/decord/issues/329
import decord  # isort: skip
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from tqdm import tqdm
import sys

import trainer.utils.distributed as dist_ops

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
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

    def get_data(self, idx):
        info = self.data_infos[idx]

        decord.bridge.set_bridge("torch")  # Load frames in torch.tensor format.
        vr = decord.VideoReader(info["video_path"])
        video = vr.get_batch(info["frame_indices"])

        video = video.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        height, width = video.shape[2:]
        scale_factor = np.sqrt(self.max_pixels / (height * width))
        new_height, new_width = round(height * scale_factor), round(width * scale_factor)
        new_height = new_height // self.spatial_downsample_factor * self.spatial_downsample_factor
        new_width = new_width // self.spatial_downsample_factor * self.spatial_downsample_factor
        if height != new_height or width != new_width:
            video = F.interpolate(video, size=(new_height, new_width), mode="bicubic", antialias=True)  # torch.uint8

        video = video.float().div_(127.5).sub_(1.0)  # [0, 255] -> [-1, 1]
        video = video.transpose(0, 1)  # [T, C, H, W] -> [C, T, H, W]

        return dict(video=video, video_path=info["video_path"], prompt=info["prompt"])
    
    def get_vace_data(self, idx):
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif')
        info = self.data_infos[idx]
        file_extension = os.path.splitext(info["video_path"])[1].lower()
        # if file_extension in image_extensions:#处理图像逻辑
        #     # 加载图像
        #     image = Image.open(os.path.abspath(info["video_path"])).convert("RGB")
        #     src_image = Image.open(os.path.abspath(info["src_video_path"])).convert("RGB")
        #     image_tensor = F_tv.to_tensor(image) # [C, H, W], float [0, 1]
        #     src_image_tensor = F_tv.to_tensor(src_image) # [C, H, W], float [0, 1]
        #     video = image_tensor.unsqueeze(0).repeat(41, 1, 1, 1)
        #     src_video = src_image_tensor.unsqueeze(0).repeat(41, 1, 1, 1)
        #     video = (video * 255).to(torch.uint8)
        #     src_video = (src_video * 255).to(torch.uint8)
        if file_extension in image_extensions: # 处理图像逻辑
            # 加载图像
            # 注意：这里假设 info["video_path"] 是原始大图，info["src_video_path"] 也是大图（或者一个占位符）
            # 如果 src_video_path 有特定用途，你需要根据实际情况调整
            original_image = Image.open(os.path.abspath(info["video_path"])).convert("RGB")
            src_image = Image.open(os.path.abspath(info["src_video_path"])).convert("RGB")

            img_width, img_height = original_image.size
            src_image = src_image.resize((img_width, img_height), Image.LANCZOS)
            if img_width > img_height:
                # 左右滑动
                slide_direction = "horizontal"
                # 裁剪尺寸为图像的短边
                crop_size = img_height
                # 可滑动距离 = 图像宽 - 裁剪尺寸
                max_slide_distance = img_width - crop_size
            else:
                # 上下滑动 (或正方形图像，也可以选择一个方向)
                slide_direction = "vertical"
                # 裁剪尺寸为图像的短边
                crop_size = img_width
                # 可滑动距离 = 图像高 - 裁剪尺寸
                max_slide_distance = img_height - crop_size

            # 如果图像是正方形或者已经无法滑动（短边等于长边），则不滑动，直接中心裁剪
            if max_slide_distance <= 0:
                # 无法滑动，直接中心裁剪1:1画面
                center_crop_transform = F_tv.CenterCrop(min(img_width, img_height))
                cropped_image = center_crop_transform(original_image)
                image_tensor = F_tv.to_tensor(cropped_image)
                src_image_tensor = F_tv.to_tensor(cropped_image) # src_image 也使用中心裁剪
                
                # 视频只有一帧，但为了保持与下面逻辑一致，repeat成多帧
                num_frames = 41 # 固定帧数
                video = image_tensor.unsqueeze(0).repeat(num_frames, 1, 1, 1)
                src_video = src_image_tensor.unsqueeze(0).repeat(num_frames, 1, 1, 1)
                video = (video * 255).to(torch.uint8)
                src_video = (src_video * 255).to(torch.uint8)
                
            else:
                # 2. 生成固定帧数的视频
                num_frames = 61 # 固定帧数，可以根据需要调整
                frames_per_slide_one_way = num_frames
                if frames_per_slide_one_way == 0: # 避免除零
                    frames_per_slide_one_way = 1

                # 每帧滑动距离 (浮点数，以便更平滑)
                slide_step_per_frame = max_slide_distance / frames_per_slide_one_way

                # 用于存储所有帧的列表
                video_frames = []
                src_video_frames = [] # 如果 src_video 需要同样的滑动效果

                # 4. 循环滑动生成帧
                current_offset = 0.0 # 当前的滑动偏移量
                slide_forward = True # 标记当前是否向前滑动

                for i in range(num_frames):
                    # 裁剪图像
                    if slide_direction == "horizontal":
                        # 计算左上角 x, y
                        x = int(current_offset)
                        y = 0
                        cropped_frame = original_image.crop((x, y, x + crop_size, y + crop_size))
                        src_cropped_frame = src_image.crop((x, y, x + crop_size, y + crop_size)) # src_video 也进行相同裁剪
                    else: # vertical
                        # 计算左上角 x, y
                        x = 0
                        y = int(current_offset)
                        cropped_frame = original_image.crop((x, y, x + crop_size, y + crop_size))
                        src_cropped_frame = src_image.crop((x, y, x + crop_size, y + crop_size)) # src_video 也进行相同裁剪
                    
                    video_frames.append(F_tv.to_tensor(cropped_frame))
                    src_video_frames.append(F_tv.to_tensor(src_cropped_frame)) # src_video 也进行相同裁剪

                    # 5. 更新滑动偏移量，实现反向滑动
                    if slide_forward:
                        current_offset += slide_step_per_frame
                        # 如果到达或超过最大滑动距离，反向滑动
                        if current_offset >= max_slide_distance:
                            current_offset = max_slide_distance # 确保不超过边界
                            slide_forward = False
                    else:
                        current_offset -= slide_step_per_frame
                        # 如果回到或低于起始位置，再次正向滑动
                        if current_offset <= 0:
                            current_offset = 0 # 确保不低于边界
                            slide_forward = True

                # 将帧列表转换为 tensor
                video = torch.stack(video_frames, dim=0) # [T, C, H, W]
                src_video = torch.stack(src_video_frames, dim=0) # [T, C, H, W]

                # 转换为 uint8 [0, 255]
                video = (video * 255).to(torch.uint8)
                src_video = (src_video * 255).to(torch.uint8)
        else:
            decord.bridge.set_bridge("torch")  # Load frames in torch.tensor format.
            #print("frame_indices", info["frame_indices"])
            vr = decord.VideoReader(info["video_path"])
            #print("video info", vr.get_avg_fps(), len(vr))
            video = vr.get_batch(info["frame_indices"])

            src_vr = decord.VideoReader(info["src_video_path"])
            #print("src video info", src_vr.get_avg_fps(), len(src_vr))
            src_video = src_vr.get_batch(info["frame_indices"])

            video = video.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
            src_video = src_video.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        
        current_height, current_width = video.shape[2:]

        if current_height != current_width: # 如果不是1:1，则进行裁剪
            min_dim = min(current_height, current_width)
            
            # 计算中心裁剪的起始坐标
            start_h = (current_height - min_dim) // 2
            start_w = (current_width - min_dim) // 2
            
            # 执行裁剪
            video = video[:, :, start_h : start_h + min_dim, start_w : start_w + min_dim]
            src_video = src_video[:, :, start_h : start_h + min_dim, start_w : start_w + min_dim]
            
            logger.debug(f"Cropped from ({current_height}x{current_width}) to ({min_dim}x{min_dim}).")

        height, width = video.shape[2:]
        scale_factor = np.sqrt(self.max_pixels / (height * width))
        new_height, new_width = round(height * scale_factor), round(width * scale_factor)
        new_height = new_height // self.spatial_downsample_factor * self.spatial_downsample_factor
        new_width = new_width // self.spatial_downsample_factor * self.spatial_downsample_factor
        # new_height = 704
        # new_width = 1280
        if height != new_height or width != new_width:
            video = F.interpolate(video, size=(new_height, new_width), mode="bicubic", antialias=True)  # torch.uint8
            src_video = F.interpolate(src_video, size=(new_height, new_width), mode="bicubic", antialias=True)  # torch.uint8

        video = video.float().div_(127.5).sub_(1.0)  # [0, 255] -> [-1, 1]
        src_video = src_video.float().div_(127.5).sub_(1.0)  # [0, 255] -> [-1, 1]
        video = video.transpose(0, 1)  # [T, C, H, W] -> [C, T, H, W]
        src_video = src_video.transpose(0, 1)  # [T, C, H, W] -> [C, T, H, W]
        assert src_video.shape==video.shape

        return dict(video=video, src_video=src_video, 
                    video_path=info["video_path"], 
                    src_video_path=info["src_video_path"],
                    prompt=info["prompt"])
    
    # def get_vace_image_data(self, idx):
    #     info = self.data_infos[idx]

    #     decord.bridge.set_bridge("torch")  # Load frames in torch.tensor format.
    #     #print("frame_indices", info["frame_indices"])
    #     vr = decord.VideoReader(info["video_path"])
    #     #print("video info", vr.get_avg_fps(), len(vr))
    #     video = vr.get_batch(info["frame_indices"])

    #     src_vr = decord.VideoReader(info["src_video_path"])
    #     #print("src video info", src_vr.get_avg_fps(), len(src_vr))
    #     src_video = src_vr.get_batch(info["frame_indices"])

    #     video = video.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
    #     src_video = src_video.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]

    #     height, width = video.shape[2:]
    #     scale_factor = np.sqrt(self.max_pixels / (height * width))
    #     new_height, new_width = round(height * scale_factor), round(width * scale_factor)
    #     new_height = new_height // self.spatial_downsample_factor * self.spatial_downsample_factor
    #     new_width = new_width // self.spatial_downsample_factor * self.spatial_downsample_factor
    #     # new_height = 704
    #     # new_width = 1280
    #     if height != new_height or width != new_width:
    #         video = F.interpolate(video, size=(new_height, new_width), mode="bicubic", antialias=True)  # torch.uint8
    #         src_video = F.interpolate(src_video, size=(new_height, new_width), mode="bicubic", antialias=True)  # torch.uint8

    #     video = video.float().div_(127.5).sub_(1.0)  # [0, 255] -> [-1, 1]
    #     src_video = src_video.float().div_(127.5).sub_(1.0)  # [0, 255] -> [-1, 1]
    #     video = video.transpose(0, 1)  # [T, C, H, W] -> [C, T, H, W]
    #     src_video = src_video.transpose(0, 1)  # [T, C, H, W] -> [C, T, H, W]
    #     assert src_video.shape==video.shape

    #     # target_length = 81
    #     # pad_value = 0.0

    #     # # 2. 处理 'video' 张量
    #     # current_t_video = video.shape[1]  # 获取第二个维度 T 的当前长度
    #     # if current_t_video < target_length:
    #     #     # 计算需要填充的长度
    #     #     padding_needed = target_length - current_t_video
    #     #     pad_tuple = (0, 0,  # 不填充 W 维度 (左右)
    #     #                 0, 0,  # 不填充 H 维度 (上下)
    #     #                 0, padding_needed) # 在 T 维度的末尾填充
                        
    #     #     # 执行填充
    #     #     video = F.pad(video, pad_tuple, "constant", pad_value)
    #     #     src_video = F.pad(src_video, pad_tuple, "constant", pad_value)
    #     #     #print(f"Padded 'src_video' from {current_t_src_video} to {src_video.shape[1]} frames.")
    #     return dict(video=video, src_video=src_video, 
    #                 video_path=info["video_path"], 
    #                 src_video_path=info["src_video_path"],
    #                 prompt=info["prompt"])

    def preprocess(self, data_path):
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif')
        with open(data_path, "r", encoding="utf-8") as f:
            data_infos = json.load(f)

        new_data_infos = []
        too_short_videos = []
        for info in data_infos:
            file_extension = os.path.splitext(info["video_path"])[1].lower()

            if file_extension in image_extensions:
                info["frame_indices"] = [0]
                new_data_infos.append(info)
                continue

            vr = decord.VideoReader(info["video_path"])
            src_vr = decord.VideoReader(info["src_video_path"])
            fps = vr.get_avg_fps()
            num_frames = len(src_vr)
            duration = num_frames / fps

            # if duration > 5.5:
            #     logger.warning(
            #         f"'{info['src_video_path']}' duration is {duration:.1f} seconds. Consider using --train_duration."
            #     )

            # Resample in case high fps, such as 50/60/90/144 -> train_fps (e.g, 16)
            load_fps = self.train_fps
            if self.train_duration is not None:
                load_fps = self.train_fps / duration * self.train_duration
            frame_interval = 1.0 if abs(fps - load_fps) < 0.1 else fps / load_fps
            frame_indices = np.arange(0, num_frames, frame_interval).astype(int)
            frame_indices = frame_indices[frame_indices < num_frames]

            # The video is too short, skip it.
            if len(frame_indices) < min(self.frame_bucket):
                too_short_videos.append(info["video_path"])
                continue

            num_frames = max([x for x in self.frame_bucket if x <= len(frame_indices)])
            frame_indices = frame_indices[:num_frames]

            info["frame_indices"] = frame_indices.tolist()
            new_data_infos.append(info)

        if len(too_short_videos) > 0:
            logger.warning(f"Filtered {len(too_short_videos)} too short videos: {too_short_videos}")
        return new_data_infos

    def cache_fn(self, idx, vae, text_encoder, cache_path, clip, task):
        data = self[idx]
        video = data["video"].cuda()  # [3, T, H, W], fp32
        prompt = data["prompt"]

        T, H, W = video.shape[-3:]

        context = text_encoder([prompt], "cuda")[0]  # [70, 4096]
        video_latent = vae.encode([video])[0]  # [16, 21, 90, 50]

        if "t2v" in task:
            save_data = {"context": context, "video_latent": video_latent}
            save_file(save_data, os.path.splitext(data["video_path"])[0] + ".safetensors")
            return

        ref_video = torch.zeros(3, T, H, W).cuda()
        if "i2v" in task:
            # TODO: Check if we need to extract clip context from unresized image.
            clip_context = clip.visual([video[:, :1]]) if clip else None  # [1, 257, 1280]
            ref_video[:, 0] = video[:, 0]
        elif "lf2v" in task and "flf2v" not in task:
            clip_context = clip.visual([video[:, -1:]]) if clip else None  # [1, 257, 1280]
            ref_video[:, -1] = video[:, -1]
        elif "flf2v" in task:
            clip_context = clip.visual([video[:, :1], video[:, -1:]]) if clip else None  # [2, 257, 1280]
            ref_video[:, 0] = video[:, 0]
            ref_video[:, -1] = video[:, -1]
        elif "kf2v" in task:
            key_index = data["key_frame_index"]
            clip_context = clip.visual([video[:, :1], video[:, key_index : key_index + 1]]) if clip else None
            ref_video[:, 0] = video[:, 0]
            ref_video[:, key_index] = video[:, key_index]

        y = vae.encode([ref_video])[0]  # [16, 21, 90, 50]

        lat_h, lat_w = video_latent.shape[-2:]
        mask = torch.ones(T, lat_h, lat_w, device="cuda")
        if "i2v" in task:
            mask[1:] = 0
        elif "lf2v" in task and "flf2v" not in task:
            mask[:-1] = 0
        elif "flf2v" in task:
            mask[1:-1] = 0
        elif "kf2v" in task:
            mask[1:key_index] = 0
            mask[key_index + 1 :] = 0
        mask = torch.concat([torch.repeat_interleave(mask[:1], repeats=4, dim=0), mask[1:]], dim=0)
        mask = mask.view(mask.shape[0] // 4, 4, lat_h, lat_w)
        mask = mask.transpose(0, 1)  # [4, 21, 90, 50]

        y = torch.concat([mask, y])  # [20, 21, 90, 50]

        save_data = {"context": context, "video_latent": video_latent, "y": y}
        if clip:
            save_data["clip_context"] = clip_context
        save_file(save_data, os.path.splitext(data["video_path"])[0] + ".safetensors")

    def calculate_or_load_dataset_cache(self, vae, text_encoder, clip, task, cache_path):
        """
        NOTE: Currently, this function only supports single-node or multi-node with shared file system.
        """
        os.makedirs(cache_path, exist_ok=True)
        rank, world_size = dist_ops.get_rank(), dist_ops.get_world_size()

        video_paths = [info["video_path"] for info in self.data_infos]
        print("len(video_paths)", len(video_paths))
        print("len(self.data_infos)", len(self), len(self.data_infos))
        cache_files = []
        for video_path in video_paths:
            filename = os.path.basename(video_path)
            video_path = os.path.join(cache_path, filename)
            cache_file = os.path.splitext(video_path)[0] + ".safetensors"
            if os.path.exists(cache_file):
                cache_files.append(cache_file)
            else:
                print(f"Cache file not found for {video_path}: expected at {cache_file}")

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
