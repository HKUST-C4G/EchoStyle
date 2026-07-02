# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config

import trainer.utils.distributed as dist_ops

from .model import WanAttentionBlock, WanModel, sinusoidal_embedding_1d


class VaceWanAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=0,
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    ##############################################################
    # Modified by wan_trainer to support gradient checkpointing. #
    ##############################################################
    # def forward(self, c, x, **kwargs):
    def forward(self, c, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        if self.block_id == 0:
            c = self.before_proj(c) + x

        # c = super().forward(c, **kwargs)
        c = super().forward(c, e, seq_lens, grid_sizes, freqs, context, context_lens)
        c_skip = self.after_proj(c)
        return c, c_skip


class BaseWanAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=None,
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id

    ##############################################################
    # Modified by wan_trainer to support gradient checkpointing. #
    ##############################################################
    # def forward(self, x, hints, context_scale=1.0, **kwargs):
    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        hints,
        context_scale=1.0,
    ):
        # x = super().forward(x, **kwargs)
        x = super().forward(x, e, seq_lens, grid_sizes, freqs, context, context_lens)
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale
        return x


class VaceWanModel(WanModel):
    @register_to_config
    def __init__(
        self,
        vace_layers=None,
        vace_in_dim=None,
        model_type="vace",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        super().__init__(
            model_type,
            patch_size,
            text_len,
            in_dim,
            dim,
            ffn_dim,
            freq_dim,
            text_dim,
            out_dim,
            num_heads,
            num_layers,
            window_size,
            qk_norm,
            cross_attn_norm,
            eps,
        )

        self.vace_layers = [i for i in range(0, self.num_layers, 2)] if vace_layers is None else vace_layers
        self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim

        assert 0 in self.vace_layers
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # blocks
        self.blocks = nn.ModuleList(
            [
                BaseWanAttentionBlock(
                    "t2v_cross_attn",
                    self.dim,
                    self.ffn_dim,
                    self.num_heads,
                    self.window_size,
                    self.qk_norm,
                    self.cross_attn_norm,
                    self.eps,
                    block_id=self.vace_layers_mapping[i] if i in self.vace_layers else None,
                )
                for i in range(self.num_layers)
            ]
        )

        # vace blocks
        self.vace_blocks = nn.ModuleList(
            [
                VaceWanAttentionBlock(
                    "t2v_cross_attn",
                    self.dim,
                    self.ffn_dim,
                    self.num_heads,
                    self.window_size,
                    self.qk_norm,
                    self.cross_attn_norm,
                    self.eps,
                    block_id=i,
                )
                for i in self.vace_layers
            ]
        )

        # vace patch embeddings
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    ####################################################################################
    # Modified by wan_trainer to support sequence parallel and gradient checkpointing. #
    ####################################################################################
    def forward_vace(self, x, vace_context, seq_len, kwargs):
        # embeddings
        # [tensor(96, 22, 138, 102),] -> [tensor([1, 5120, 22, 69, 51]),]
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        # [tensor([1, 77418, 5120]),]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        # Pad sequence for ulysses: [1, 77418, 5120]
        c = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in c])

        if self.enable_sp:
            sp_group = dist_ops.get_sequence_parallel_group()
            c = dist_ops.split_forward_gather_backward(c, dim=1, group=sp_group, grad_scale="down")

        # arguments
        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)

        hints = []
        if self.enable_gc:
            for block in self.vace_blocks:
                c, c_skip = torch.utils.checkpoint.checkpoint(
                    block,
                    c,
                    x,
                    new_kwargs["e"],
                    new_kwargs["seq_lens"],
                    new_kwargs["grid_sizes"],
                    new_kwargs["freqs"],
                    new_kwargs["context"],
                    new_kwargs["context_lens"],
                    use_reentrant=False,
                )
                hints.append(c_skip)
        else:
            for block in self.vace_blocks:
                c, c_skip = block(c, **new_kwargs)
                # [1, 77418, 5120]
                hints.append(c_skip)
        return hints

    ####################################################################################
    # Modified by wan_trainer to support sequence parallel and gradient checkpointing. #
    ####################################################################################
    def forward(
        self,
        x,
        t,
        vace_context,
        context,
        seq_len,
        vace_context_scale=1.0,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # if self.model_type == 'i2v':
        #     assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # if y is not None:
        #     x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        # [tensor([16, 22, 138, 102]),] -> [tensor([1, 5120, 22, 69, 51]),]
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # value = tensor([[22, 69, 51]])
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        # [tensor([1, 77418, 5120]),]
        x = [u.flatten(2).transpose(1, 2) for u in x]
        # value = tensor([77418])
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        # Pad sequence for ulysses: [1, 77418, 5120]
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        # time embeddings
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            # [1, 6, 5120]
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        # [tensor([512, 5120]),]
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )

        # if clip_fea is not None:
        #     context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        #     context = torch.concat([context_clip, context], dim=1)

        if self.enable_sp:
            sp_group = dist_ops.get_sequence_parallel_group()
            x = dist_ops.split_forward_gather_backward(x, dim=1, group=sp_group, grad_scale="down")

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
        )

        hints = self.forward_vace(x, vace_context, seq_len, kwargs)

        if self.enable_gc:
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    e0,
                    seq_lens,
                    grid_sizes,
                    self.freqs,
                    context,
                    context_lens,
                    hints,
                    vace_context_scale,
                    use_reentrant=False,
                )
        else:
            kwargs["hints"] = hints
            kwargs["context_scale"] = vace_context_scale

            for block in self.blocks:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        if self.enable_sp:
            x = dist_ops.gather_forward_split_backward(x, dim=1, group=sp_group, grad_scale="up")

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]
