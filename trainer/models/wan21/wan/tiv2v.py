# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.distributed as dist
from tqdm import tqdm

from trainer.models.wan21.wan.configs import WAN_CONFIGS
from trainer.utils import load_lora

from .distributed.fsdp import shard_model
from .modules import echostyle as WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

def _encode_frames(frames, ref_images, masks, vae):
    if ref_images is None:
        ref_images = [None] * len(frames)
    else:
        assert len(frames) == len(ref_images)

    if masks is None:
        latents = vae.encode(frames)
        print("frame.shape:", frames[0].shape,"latents.shape:", latents[0].shape)
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
            print("latent.shape after concat ref:", latent.shape)
        cat_latents.append(latent)
        print("cat_latents len:", len(cat_latents), "cat_latents[0].shape:", cat_latents[0].shape)
    return cat_latents

class echostylemodel:
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        lora_path=None,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint), device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )

            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()

        if lora_path:
            load_lora(self.model, lora_path)

        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(
        self,
        input_prompt,
        input_frames,
        input_ref_images,
        size=(1280, 720),
        frame_num=81,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=50,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
    ):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (
            self.vae.model.z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2],
        )

        seq_len = (
            math.ceil(
                (target_shape[2] * target_shape[3])
                / (self.patch_size[1] * self.patch_size[2])
                * (target_shape[1]+1)
                / self.sp_size
            )
            * self.sp_size
        )
        #print("sp_size:", self.sp_size)
        seq_len = seq_len * 2
        #print("target_shape:", target_shape, "seq_len:", seq_len)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device("cpu"))
            context_null = self.text_encoder([n_prompt], torch.device("cpu"))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1]+1,
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g,
            )
        ]
        ref_latent = _encode_frames([input_frames], [input_ref_images], None, self.vae)[0]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, "no_sync", noop_no_sync)
        #print("noise_dtype:", noise[0].dtype, "ref_latent dtype:", ref_latent[0].dtype)

        # evaluation mode
        with torch.amp.autocast("cuda", dtype=self.param_dtype), torch.no_grad(), no_sync():
            if sample_solver == "unipc":
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False
                )
                sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == "dpm++":
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False
                )
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise
            #print("latents[0].shape:", latents[0].shape)

            arg_t_c = {"context": context, "seq_len": seq_len, "y": [ref_latent]}
            arg_null_c = {"context": context_null, "seq_len": seq_len, "y": [ref_latent]}
            arg_t_null = {"context": context, "seq_len": seq_len, "y": None}
            arg_null_null = {"context": context_null, "seq_len": seq_len, "y": None}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_t_c)[0]
                noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0), return_dict=False, generator=seed_g
                )[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            x0 = [x[:, 1:, :, :] for x in x0]  # remove ref frame padding
            #print("x0[0].shape:", x0[0].shape)
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

# if __name__ == "__main__":
#     cfg = WAN_CONFIGS["t2v-1.3B"]
#     model = echostylemodel(
#         config=cfg,
#         checkpoint_dir="../models/Wan2.1-T2V-1.3B/",
#         lora_path="training_outputs/ghibli_wan-t2v-1.3B/checkpoint-10000/model_state.pth"
#     )
    
#     expected_dtype = model.param_dtype
#     print(f"Model expects dtype: {expected_dtype}")

#     a = torch.randn(3, 81, 720, 1280, dtype=expected_dtype)
#     ref = [torch.randn(3, 1, 720, 1280, dtype=expected_dtype)]
    
#     # 确保 Tensor 在正确的设备上
#     device = model.device
#     a = a.to(device)
#     ref = [r.to(device) for r in ref]
#     print("model.device:", model.device)
#     prompt = ""
    
#     # 调用 generate
#     c = model.generate(
#         input_prompt=prompt,
#         input_frames=a,
#         input_ref_images=ref,
#         size=(1280, 720),
#         frame_num=81,
#         shift=5.0,
#         sample_solver="unipc",
#         sampling_steps=5
#     )