import json
import logging
import math
import os
import random
import shutil
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import decord

import trainer.utils.distributed as dist_ops
from trainer.datasets import Wanexv2vDataset, Wanv2vDataset
#from trainer.models.wan21.wan.configs import WAN_CONFIGS as WAN21_CONFIGS
#from trainer.models.wan21.wan.modules import CLIPModel, T5EncoderModel, VaceWanModel, WanVAE
from trainer.models.wan22.wan.modules import T5EncoderModel, Wan2_1_VAE
from trainer.models.wan22.wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from trainer.models.wan22.wan.utils.utils import save_video
from trainer.models.wan22.wan.configs import WAN_CONFIGS as WAN22_CONFIGS
from trainer.models.wan22.wan.modules import WanModel as Wan22Model
from trainer.utils import (
    Reporter,
    load_checkpoint,
    move_models_to_device,
    pad_videos_to_same_size,
    parse_args,
    save_checkpoint,
    set_random_seed,
    setup_logger,
)

logger = logging.getLogger(__name__)

def count_parameters(model: nn.Module):
    """
    计算并打印模型中可训练和不可训练参数的详细信息。
    """
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    
    print("-" * 50)
    print(f"{'Parameter Name':<30} {'Shape':<15} {'Trainable'}")
    print("-" * 50)

    for name, param in model.named_parameters():
        total_params += param.numel()
        is_trainable = param.requires_grad
        
        if is_trainable:
            trainable_params += param.numel()
        else:
            non_trainable_params += param.numel()
            
        print(f"{name:<30} {str(list(param.shape)):<15} {is_trainable}")
        
    print("-" * 50)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"不可训练 (冻结) 参数数量: {non_trainable_params:,}")
    print("-" * 50)


class NoisyModelInputGetter:
    def __init__(self, weighting_scheme="logit_normal", shift=1.0):
        self.weighting_scheme = weighting_scheme
        self.shift = shift

        if weighting_scheme != "logit_normal_continuous":
            self.noise_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False
            )

    def compute_density_for_timestep_sampling(
        self,
        weighting_scheme: str,
        batch_size: int,
        logit_mean: float = None,
        logit_std: float = None,
        mode_scale: float = None,
    ):
        """
        Compute the density for sampling the timesteps when doing SD3 training.

        Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

        SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
        """
        if weighting_scheme == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
            u = torch.nn.functional.sigmoid(u)
        elif weighting_scheme == "mode":
            u = torch.rand(size=(batch_size,), device="cpu")
            u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        else:
            u = torch.rand(size=(batch_size,), device="cpu")
        return u

    def compute_loss_weighting_for_sd3(self, weighting_scheme: str, sigmas=None):
        """
        Computes loss weighting scheme for SD3 training.

        Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

        SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
        """
        if weighting_scheme == "sigma_sqrt":
            weighting = (sigmas**-2.0).float()
        elif weighting_scheme == "cosmap":
            bot = 1 - 2 * sigmas + 2 * sigmas**2
            weighting = 2 / (math.pi * bot)
        else:
            weighting = torch.ones_like(sigmas)
        return weighting

    def get_noisy_model_input(self, video_latent, noise, min_timestep, max_timestep):
        if self.weighting_scheme != "logit_normal_continuous":
            while True:
                # Sample a random timestep for weighting schemes where we sample timesteps non-uniformly.
                u = self.compute_density_for_timestep_sampling(
                    weighting_scheme=self.weighting_scheme,
                    batch_size=1,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
                indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
                sigmas = self.noise_scheduler.sigmas[indices][None, None, None].to("cuda")  # shape: [1, 1, 1, 1]
                timesteps = self.noise_scheduler.timesteps[indices].to("cuda")

                if timesteps.item() >= min_timestep and timesteps.item() <= max_timestep:
                    break

            # Add noise according to flow matching.
            # zt = (1 - texp) * x + texp * z1
            noisy_model_input = (1.0 - sigmas) * video_latent + sigmas * noise

            # These weighting schemes use a uniform timestep sampling and instead post-weight the loss.
            weighting = self.compute_loss_weighting_for_sd3(weighting_scheme=self.weighting_scheme, sigmas=sigmas)
        else:
            while True:
                t = torch.randn(1, device=video_latent.device).sigmoid()
                t = (t * self.shift) / (1 + (self.shift - 1) * t)
                timesteps = t * 1000.0

                if timesteps >= min_timestep and timesteps <= max_timestep:
                    break

            t = t.view(1, 1, 1, 1)
            noisy_model_input = (1 - t) * video_latent + t * noise
            weighting = torch.ones_like(t)

        return noisy_model_input, timesteps, weighting


def main(args):
    # Initialize distributed environment.
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # sp_size = args.sequence_parallel_size if args.enable_sequence_parallel else 1
    sp_size = 8 if args.enable_fsdp else 1
    setup_logger(rank=rank, output_dir=args.output_dir)  # Log only on master process.
    dist_ops.init_process_groups(world_size, sp_size)

    # Set random seed for reproducibility. If no seed provided, use randomly generated seed.
    set_random_seed(args.seed + rank if args.seed and args.seed >= 0 else None)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # Handle multiple kinds of reporters.
    reporter = Reporter(args)

    is_wan22 = args.task in WAN22_CONFIGS
    print("Not using WAN22.") if not is_wan22 else print("Using WAN22.")
    cfg = WAN22_CONFIGS[args.task]

    vae = Wan2_1_VAE(vae_pth=os.path.join(args.pretrained, cfg.vae_checkpoint), device="cuda")

    text_encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,  # bf16
        device="cuda",
        checkpoint_path=os.path.join(args.pretrained, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(args.pretrained, cfg.t5_tokenizer),
        shard_fn=None,
    )

    clip = None
    if is_wan22:
        model = Wan22Model.from_pretrained(args.pretrained, subfolder=f"{args.wan22_train_part}_model")
        timestep_config_key = args.wan22_train_part + "_t2v" if "t2v" in args.task else args.wan22_train_part
        min_timestep, max_timestep = {
            "high_noise_t2v": (875, 1000),
            "low_noise_t2v": (0, 875),
            "high_noise": (900, 1000),
            "low_noise": (0, 900),
            "high_noise_1": (970, 1000),
            "high_noise_2": (900, 970),
            "low_noise_3": (800, 900),
            "low_noise_4": (0, 800),
        }[timestep_config_key]
        print(f"Using WAN2.2 {args.wan22_train_part} model with timestep range: {min_timestep} to {max_timestep}.")
    else:
        print("Error: Only WAN2.2 is supported in this script.")
        return

    if args.enable_lora:
        logger.info("Enable LoRA training.")
        model.requires_grad_(False)

        # For mixed precision training, we cast all non-trainable weights excluding vae (text_encoder, clip and
        # non-lora dit) to half-precision, as these weights are only used for inference, keeping weights in full
        # precision is not required.
        # NOTE: vae is always fp32. text_encoder and clip are already half-precision.
        model.to(dtype=torch.bfloat16)

        # Now we will add new LoRA weights to the model.
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=[name for name, module in model.named_modules() if isinstance(module, nn.Linear)],
        )
        # `hf_device_map` specifies which submodule of the model should be run on which device.
        # If not set to None, we will encounter device mismatch error during evaluation.
        # TODO: Figure out why training is not affected by this.
        model.hf_device_map = None
        model = get_peft_model(model, lora_config)
        trainable_params, all_params = model.get_nb_trainable_parameters()
        logger.info(
            f"trainable params: {trainable_params:,d} || all params: {all_params:,d} || "
            f"trainable%: {100 * trainable_params / all_params:.4f}"
        )
    else:
        assert next(model.parameters()).dtype == torch.float32, "Model parameters must be fp32."
        count_parameters(model)
        #trainable_params, all_params = model.get_nb_trainable_parameters()

    if args.enable_gradient_checkpointing:
        model.enable_gradient_checkpointing()

    if args.enable_sequence_parallel:
        model.enable_sequence_parallel()

    if args.enable_fsdp:
        logger.info("Enable FSDP.")
        # In general, we hope that the trainable weights (LoRA) are in fp32 to avoid rounding errors.
        # However, FSDP requires the weights to have a uniform dtype, so we convert the model to bf16.
        model = FSDP(
            module=model.to(torch.bfloat16),
            device_mesh=dist_ops.get_device_mesh(),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks),
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
            ),
            device_id=local_rank,
            use_orig_params=True,
        )
    else:
        model = DDP(model.cuda(), device_ids=[local_rank])

    # Training summary:
    # +--------------+----------------+---------------------+------------------+---------------+
    # |    model     | autocast dtype | model weights dtype |  requires_grad   | is train mode |
    # +--------------+----------------+---------------------+------------------+---------------+
    # |     vae      |      fp32      |         fp32        |      False       |     False     |
    # | text_encoder |      None      |         bf16        |      False       |     False     |
    # |     clip     |      fp16      |         fp16        |      False       |     False     |
    # |     dit      |      bf16      |   bf16+fp32(lora)   | False+True(lora) |      True     |
    # |   dit-fsdp   |      bf16      |         bf16        |       True       |      True     |
    # +--------------+----------------+---------------------+------------------+---------------+

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )

    # Build dataset.
    # train_dataset = Wanv2vDataset(
    #     data_path=args.data_path,
    #     frame_bucket=args.frame_bucket,
    #     train_fps=args.train_fps,
    #     train_duration=args.train_duration,
    #     max_pixels=args.max_pixels,
    #     spatial_downsample_factor=args.spatial_downsample_factor,
    # )
    train_dataset = Wanv2vDataset(
        data_path=args.data_path,
        frame_bucket=args.frame_bucket,
        train_fps=args.train_fps,
        train_duration=args.train_duration,
        max_pixels=args.max_pixels,
        spatial_downsample_factor=args.spatial_downsample_factor,
    )

    # Calculate the dataset cache distributedly.
    train_dataset.calculate_or_load_dataset_cache(vae, text_encoder, clip, args.task, args.cache_path)
    move_models_to_device(text_encoder, clip, device="cpu")

    # Ensure that processes in the same sequence parallel group read the same data.
    dp_world_size = dist_ops.get_data_parallel_world_size()
    dp_rank = dist_ops.get_data_parallel_rank()
    train_sampler = DistributedSampler(train_dataset, num_replicas=dp_world_size, rank=dp_rank)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=False,  # DistributedSampler will do shuffle, so we set `shuffle=False`
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Constant LR scheduler, as we are not sure how long we need to train.
    lr_scheduler = LambdaLR(optimizer, lambda _: 1, last_epoch=-1)

    # Priority: max_train_steps > num_train_epochs
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = args.train_batch_size * world_size * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Potentially load in the weights and states from a previous save.
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.warning(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            load_checkpoint(model, optimizer, os.path.join(args.output_dir, path), args.enable_fsdp)
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # Show the progress bar once on each machine.
    disable = not local_rank == 0
    progress_bar = tqdm(range(0, args.max_train_steps), initial=initial_global_step, desc="Steps", disable=disable)
    # Ensure that processes in the same sequence parallel group initialize the same random noise.
    generator = torch.Generator("cuda").manual_seed(global_step + rank)
    noisy_model_input_getter = NoisyModelInputGetter(args.weighting_scheme, args.flow_shift)
    micro_step = 0  # Track gradient accumulation steps.
    accumulated_loss = 0  # Accumulate loss across gradient accumulation steps.
    for epoch in range(first_epoch, args.num_train_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        #修改数据集的输出格式，计算video_latent和context
        for batch in train_dataloader:
            video_latent = batch["video_latent"][0].cuda()
            lat_t, lat_h, lat_w = video_latent.shape[-3:]
            set_zero = torch.rand(1).item()
            if set_zero >= 0.2:
                src_context = batch["src_context"][0].cuda()
                msk = torch.zeros((4, lat_t, lat_h, lat_w), device=src_context.device, dtype=src_context.dtype)
            else:
                src_context = batch["tep_context"][0].cuda()
                msk = torch.zeros(1, (lat_t-1)*4+1, lat_h, lat_w, device=src_context.device, dtype=src_context.dtype)
                msk[:, :16] = 1.0
                msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
                msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
                msk = msk.transpose(1, 2)[0]
            #print("video_latent shape:", video_latent.shape)
            # set_zero = torch.rand(1).item()
            #set_zero = 0.2   
            # if set_zero <= 0.1: #0.2概率，进行第一段latent引导的生成，用于长视频拓展
            #     src_context = torch.zeros_like(src_context).cuda()
            # else: 30%概率不做任何修改,进行长视频中，首段视频生成
                
            if random.random() < 1.0: # 70% 的概率
                context = batch["context"][0].cuda()
                #print("使用完整prompt")
            else: # 30% 的概率
                context = batch["r_context"][0].cuda()
                #print("使用粗略prompt")

            # video1 = vae.decode([src_context])[0]
            # video2 = vae.decode([video_latent])[0]  # 预热vae解码器
            # output_path1 = os.path.join("./debug_videos", f"epoch{epoch}_step{global_step}_rank{rank}_src.mp4")
            # output_path2 = os.path.join("./debug_videos", f"epoch{epoch}_step{global_step}_rank{rank}_video.mp4")
            # save_video(
            #     tensor=video1[None],
            #     save_file=output_path1,
            #     fps=cfg.sample_fps,
            #     nrow=1,
            #     normalize=True,
            #     value_range=(-1, 1),
            #     )
            # save_video(
            #     tensor=video2[None],
            #     save_file=output_path2,
            #     fps=cfg.sample_fps,
            #     nrow=1,
            #     normalize=True,
            #     value_range=(-1, 1),
            #     )
            #print("context shape:", context.shape)
            #src_context = batch["src_context"][0].cuda()
            #clip_context = batch["clip_context"][0].cuda() if clip else None
            #y = batch["y"][0].cuda()
            #msk = torch.zeros((4, lat_t, lat_h, lat_w), device=src_context.device, dtype=src_context.dtype)
            y = torch.cat([msk, src_context], dim=0)
            #print("y shape:", y.shape)

            seq_len = lat_t * lat_h * lat_w // 4  # dit patch size = 4
            seq_len = math.ceil(seq_len / sp_size) * sp_size

            # Sample noise that we'll add to the latents.
            noise = torch.randn(video_latent.shape, device="cuda", dtype=torch.float32, generator=generator)

            noisy_model_input, timesteps, weighting = noisy_model_input_getter.get_noisy_model_input(
                video_latent, noise, min_timestep, max_timestep
            )

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model_kwargs = dict(x=[noisy_model_input], t=timesteps, context=[context], seq_len=seq_len)
                if is_wan22:
                    model_kwargs.update(y=[y])
                else:
                    print("Error: Only WAN2.2 is supported in this script.")
                model_pred = model(**model_kwargs)[0]
            # Flow matching loss.
            target = noise - video_latent

            # Compute regular loss.
            loss = (weighting.float() * (model_pred.float() - target.float()) ** 2).mean()
            loss = loss / args.gradient_accumulation_steps
            #print("loss:", loss.item())

            # Gather the losses across all processes for logging (if we use distributed training).
            reduced_loss = loss.detach().clone()
            dist.all_reduce(reduced_loss)
            accumulated_loss += reduced_loss.item() / world_size

            micro_step += 1
            # finish_accumulated_step = micro_step % args.gradient_accumulation_steps == 0

            # Backward and optimization.
            loss.backward()
            del video_latent, context, noise, noisy_model_input, timesteps, weighting, model_pred, target, loss, reduced_loss
            torch.cuda.empty_cache()
            #print("loss:", loss.item())
            # if not finish_accumulated_step:
            #     continue
            # if args.max_grad_norm > 0:
            #     if args.enable_fsdp:
            #         model.clip_grad_norm_(args.max_grad_norm)
            #     else:
            #         clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            global_step += 1
            progress_bar.update(1)
            lr = lr_scheduler.get_last_lr()[0]
            progress_bar.set_postfix(loss=accumulated_loss, lr=lr)
            if rank == 0:
                print(f"Step: {global_step}, Loss: {accumulated_loss:.4f}, LR: {lr:.6f}")
            reporter.log_loss(accumulated_loss, lr, global_step)
            accumulated_loss = 0

            if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps:
                # Before saving state, check if this save would set us over the `checkpoints_total_limit`
                if rank == 0 and args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # Before we save the new checkpoint, we need to have at most
                    # `checkpoints_total_limit - 1` checkpoints
                    num_ckpts = len(checkpoints)
                    if num_ckpts >= args.checkpoints_total_limit:
                        num_to_remove = num_ckpts - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(f"{num_ckpts} checkpoints already exist, removing {num_to_remove} checkpoints")
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                try:
                    save_checkpoint(model, optimizer, save_path, args.enable_fsdp, rank)
                    logger.info(f"Saved checkpoint at step {global_step} to {save_path}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint at step {global_step} to {save_path}: {e}")

            if (
                args.validation_data_file is not None
                and global_step >= args.validation_start_step
                and global_step % args.validation_steps == 0
            ):
                validate(text_encoder, vae, clip, model, global_step, dp_rank, local_rank, dp_world_size, reporter)

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
