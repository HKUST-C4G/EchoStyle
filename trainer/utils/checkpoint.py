import logging
import os

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

logger = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, output_dir, fsdp=False, rank=0, save_trainable=True):
    # I don't know how to make `PeftModel.save_pretrained` work with FSDP.
    # So we save the model's state_dict directly.
    logger.info(f"Saving checkpoint to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    if save_trainable:
        with FSDP.summon_full_params(model, writeback=False, rank0_only=True, offload_to_cpu=True):
            trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]

    if fsdp:
        state_config = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        optim_config = FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True)
        # TODO: `FSDP.state_dict_type` is being deprecated. Replace it with Distributed Checkpoint (DCP).
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_config, optim_config):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if rank == 0:
        model_state = {k: v for k, v in model_state.items() if not save_trainable or k in trainable_params}
        torch.save(model_state, os.path.join(output_dir, "model_state.pth"))
        torch.save(optim_state, os.path.join(output_dir, "optimizer_state.pth"))


def load_checkpoint(model, optimizer, checkpoint_dir, fsdp=False):
    logger.info(f"Loading checkpoint from {checkpoint_dir}...")

    model_state = torch.load(os.path.join(checkpoint_dir, "model_state.pth"), weights_only=True)
    optim_state = torch.load(os.path.join(checkpoint_dir, "optimizer_state.pth"), weights_only=True)

    def load_and_check(model, state_dict):
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        lora_missing_keys = [k for k in missing_keys if "lora" in k]
        assert len(lora_missing_keys) == 0, f"Missing LoRA keys: {lora_missing_keys}"
        assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"

    if fsdp:
        state_config = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        optim_config = FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_config, optim_config):
            load_and_check(model, model_state)
            optim_state = FSDP.optim_state_dict_to_load(model, optimizer, optim_state)
            optimizer.load_state_dict(optim_state)
    else:
        load_and_check(model, model_state)
        optimizer.load_state_dict(optim_state)


def load_lora(model, lora_path, lora_rank=32):
    logger.info(f"Loading LoRA from: {lora_path}")
    model.hf_device_map = None
    target_modules = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    lora_config = LoraConfig(r=lora_rank, lora_alpha=32, target_modules=target_modules)
    model = get_peft_model(model, lora_config)
    model.requires_grad_(False)  # FSDP requires uniform requires_grad.
    if os.path.isdir(lora_path):
        lora_state_dict = load_file(os.path.join(lora_path, "adapter_model.safetensors"))
        new_lora_state_dict = {}
        for k, v in lora_state_dict.items():
            k = k.replace("lora_A", "lora_A.default").replace("lora_B", "lora_B.default")
            new_lora_state_dict[k] = v
    else:
        assert ".pth" in lora_path, f"Invalid LoRA path: {lora_path}"
        lora_state_dict = torch.load(lora_path, weights_only=True)
        new_lora_state_dict = {k.replace("module.", ""): v for k, v in lora_state_dict.items()}
    missing_keys, unexpected_keys = model.load_state_dict(new_lora_state_dict, strict=False)
    lora_missing_keys = [k for k in missing_keys if "lora" in k]
    assert len(lora_missing_keys) == 0, f"Missing lora keys: {lora_missing_keys}"
    assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
