import argparse
import json
import os


def check_validation_file(validation_data_file, task):
    assert os.path.exists(validation_data_file), f"Validation data file {validation_data_file} does not exist."

    try:
        with open(validation_data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"Validation data file {validation_data_file} is not a valid JSON file.")

    assert isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict), (
        f"{task.upper()} validation data must be a non-empty list of dictionaries."
    )

    # I2V and FLF2V share the same format: [dict(first_frame_path="xxx", last_frame_path="xxx", prompt="xxx"), ...]
    # for d in data:
    #     if task == "vace":
    #         # At least one of first_frame_path, key_frame_path, last_frame_path, and vace_ref_image_paths must exist.
    #         keys = ["first_frame_path", "key_frame_path", "last_frame_path", "vace_ref_image_paths"]
    #         found = False
    #         for key in keys:
    #             if key in d:
    #                 found = True
    #                 if key == "vace_ref_image_paths":
    #                     for img_path in d[key]:
    #                         assert os.path.exists(img_path), f"VACE ref image {img_path} does not exist."
    #                 else:
    #                     assert os.path.exists(d[key]), f"{key} {d[key]} does not exist."
    #         assert found, f"VACE validation data must contain at least one of {keys}."
    #     else:
    #         # Check the shared fields of I2V and FLF2V.
    #         assert "first_frame_path" in d and "prompt" in d, (
    #             f"{task.upper()} validation data must contain 'first_frame_path' and 'prompt' keys."
    #         )
    #         assert os.path.exists(d["first_frame_path"]), f"First frame {d['first_frame_path']} does not exist."

    #         # FLF2V requires additional last_frame_path.
    #         if task == "flf2v":
    #             assert "last_frame_path" in d, "FLF2V validation data must contain 'last_frame_path' key."
    #             assert os.path.exists(d["last_frame_path"]), f"Last frame {d['last_frame_path']} does not exist."


def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--pretrained", type=str, required=True, help="Path to the pretrained Wan2.1 / Wan2.2 model.")
    parser.add_argument("--enable_lora", action="store_true", help="Enable LoRA.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha of LoRA.")
    parser.add_argument(
        "--spatial_downsample_factor",
        type=int,
        default=16,
        help="Spatial downsample factor. Default to 16, which is 8 (vae) * 2 (patchfy).",
    )

    # Data
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to a directory containing (video, text) pairs."
    )
    parser.add_argument(
        "--cache_path", type=str, required=True, help="Path to save cache files."
    )
    parser.add_argument(
        "--frame_bucket",
        nargs="+",
        type=int,
        default=[81],
        help="Multiple possible video lengths. Video will be assigned to the maximum length in the frame_bucket.",
    )
    parser.add_argument(
        "--train_fps", type=int, default=16, help="Regardless of the fps of the raw data, always train at a fixed fps."
    )
    parser.add_argument(
        "--train_duration",
        type=float,
        default=None,
        help="If provided, train at a fixed duration by speeding up the video.",
    )
    parser.add_argument(
        "--max_pixels", type=int, default=1280 * 720, help="Maximum height * width for video resolution."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )

    # Training
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--enable_gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm. If > 0, gradient clipping will be applied.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=1000, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "logit_normal_continuous", "mode", "cosmap"],
        help=(
            "sigma_sqrt/logit_normal/mode/cosmap: diffusion weight is obtained from noise_scheduler.sigmas, "
            "logit_normal_continuous: diffusion weight is obtained from random sampling."
        ),
    )
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=1.0,
        help="A very IMPORTANT parameter to determine how to sample training timestep.",
    )
    parser.add_argument(
        "--wan22_train_part",
        type=str,
        default="high_noise",
        choices=["high_noise", "low_noise", "high_noise_1", "high_noise_2", "low_noise_3", "low_noise_4"],
        help="The part of the Wan2.2 model to train.",
    )

    # Parallelism
    parser.add_argument("--enable_sequence_parallel", action="store_true", help="Enable sequence parallel.")
    parser.add_argument(
        "--sequence_parallel_size", type=int, default=1, help="Number of GPUs used to parallelize sequence."
    )
    parser.add_argument(
        "--enable_fsdp",
        action="store_true",
        help="Enable FSDP. FSDP shares the same process group with sequence parallel.",
    )

    # Output & Logging
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard / Wandb log directory inside the output_dir. Default to 'logs'.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        choices=["tensorboard", "wandb"],
        help="The integration to report the results and logs to. If None, we save validation results to disk.",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=200,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit", type=int, default=None, help="Max number of checkpoints to store."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            " `--checkpointing_steps`, or `'latest'` to automatically select the last available checkpoint."
        ),
    )

    # Validation
    parser.add_argument(
        "--validation_data_file",
        type=str,
        default=None,
        help="Path to a json file containing validation data. Format: {'image_path_1': 'prompt_1', 'image_path_2': 'prompt_2'}",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Validate the model every X updates. `validation_data_file` must be provided.",
    )
    parser.add_argument(
        "--validation_start_step",
        type=int,
        default=0,
        help="Start validation after this many steps. Only validate when global_step >= validation_start_step.",
    )

    # Misc
    parser.add_argument(
        "--task",
        type=str,
        default="i2v-14B",
        choices=[
            "t2v-1.3B",
            "i2v-14B",
            "ti2v-5B",
            "flf2v-14B",
            "vace-14B",
            "vace-1.3B",
            "t2v-A14B",
            "i2v-A14B",
            "lf2v-A14B",
            "flf2v-A14B",
            "kf2v-A14B",
            "i2v-A14B-flash",
            "lf2v-A14B-flash",
            "flf2v-A14B-flash",
            "kf2v-A14B-flash",
        ],
        help="Task to train on.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )

    args = parser.parse_args()

    if "A14B" in args.task:
        assert args.validation_data_file is None, "Evaluation is not supported for Wan2.2 models."
    if args.validation_data_file is not None:
        check_validation_file(args.validation_data_file, args.task)
    if args.enable_fsdp:
        # if not args.enable_sequence_parallel:
        #     raise ValueError("FSDP requires sequence parallel to be enabled.")
        if args.gradient_accumulation_steps > 1:
            raise ValueError("FSDP does not support gradient accumulation.")

    return args
