import math
from typing import Union, List

from einops import repeat
from jaxtyping import Bool, Float, Int
import torch
from torch import nn
import torch.nn.functional as F
from x_transformers.x_transformers import RotaryEmbedding

from rift_svc.modules import (
    AdaLayerNormZero_Final,
    DiTBlock,
    TimestepEmbedding,
    LoRALinear,
)
 
# Conditional embedding for f0, rms, cvec
class CondEmbedding(nn.Module):
    def __init__(self, cvec_dim: int, cond_dim: int):
        super().__init__()
        self.cvec_dim = cvec_dim
        self.cond_dim = cond_dim

        self.f0_embed = nn.Linear(1, cond_dim)
        self.rms_embed = nn.Linear(1, cond_dim)
        self.cvec_embed = nn.Linear(cvec_dim, cond_dim)
        self.out = nn.Linear(cond_dim, cond_dim)

        self.ln_cvec = nn.LayerNorm(cond_dim, elementwise_affine=False, eps=1e-6)
        self.ln = nn.LayerNorm(cond_dim, elementwise_affine=True, eps=1e-6)


    def forward(
            self,
            f0: Float[torch.Tensor, "b n"],
            rms: Float[torch.Tensor, "b n"],
            cvec: Float[torch.Tensor, "b n d"],
        ):
        if f0.ndim == 2:
            f0 = f0.unsqueeze(-1)
        if rms.ndim == 2:
            rms = rms.unsqueeze(-1)

        f0_embed = self.f0_embed(f0 / 1200)
        rms_embed = self.rms_embed(rms)
        cvec_embed = self.ln_cvec(self.cvec_embed(cvec))

        cond = f0_embed + rms_embed + cvec_embed
        cond = self.ln(self.out(cond))
        return cond


# noised input audio and context mixing embedding
class InputEmbedding(nn.Module):
    def __init__(self, mel_dim: int, out_dim: int):
        super().__init__()
        self.mel_embed = nn.Linear(mel_dim, out_dim)
        self.proj = nn.Linear(2 * out_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: Float[torch.Tensor, "b n d1"], cond_embed: Float[torch.Tensor, "b n d2"]):
        x = self.mel_embed(x)
        x = torch.cat((x, cond_embed), dim = -1)
        x = self.proj(x)
        x = self.ln(x)
        return x


# backbone using DiT blocks
class DiT(nn.Module):
    def __init__(self,
                 dim: int, depth: int, head_dim: int = 64, dropout: float = 0.0, ff_mult: int = 4,
                 n_mel_channels: int = 128, num_speaker: int = 1, cvec_dim: int = 768, 
                 kernel_size: int = 31, zero_null_spk: bool = False,
                 init_std: float = 1):
        super().__init__()
    
        self.num_speaker = num_speaker
        self.spk_embed = nn.Embedding(num_speaker, dim)
        self.null_spk_embed = nn.Embedding(1, dim)
        self.tembed = TimestepEmbedding(dim)
        self.cond_embed = CondEmbedding(cvec_dim, dim)   
        self.input_embed = InputEmbedding(n_mel_channels, dim)

        self.rotary_embed = RotaryEmbedding(head_dim)

        self.dim = dim
        self.depth = depth
        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim = dim,
                    head_dim = head_dim,
                    ff_mult = ff_mult,
                    dropout = dropout,
                    kernel_size = kernel_size,
                )
                for _ in range(depth)
            ]
        )

        self.norm_out = AdaLayerNormZero_Final(dim)
        self.output = nn.Linear(dim, n_mel_channels)

        self.init_std = init_std
        self.apply(self._init_weights)
        for block in self.transformer_blocks:
            torch.nn.init.constant_(block.attn_norm.proj.weight, 0)
            torch.nn.init.constant_(block.attn_norm.proj.bias, 0)

        torch.nn.init.constant_(self.norm_out.proj.weight, 0)
        torch.nn.init.constant_(self.norm_out.proj.bias, 0)
        torch.nn.init.constant_(self.output.weight, 0)
        torch.nn.init.constant_(self.output.bias, 0)

        if zero_null_spk:
            self.null_spk_embed.weight.data.zero_()
            self.null_spk_embed.requires_grad = False

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            fan_out, fan_in = module.weight.shape
            # Spectral parameterization from the [paper](https://arxiv.org/abs/2310.17813).
            init_std = (self.init_std / math.sqrt(fan_in)) * min(1, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            # weight shape: (out_channels, in_channels/groups, kernel_size)
            fan_out = module.weight.shape[0]  # out_channels
            fan_in = module.weight.shape[1] * module.weight.shape[2]  # (in_channels/groups) * kernel_size
            init_std = (self.init_std / math.sqrt(fan_in)) * min(1, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.init_std/math.sqrt(self.dim))


    def forward(
        self,
        x: Float[torch.Tensor, "b n d1`"],  # nosied input mel
        spk: Int[torch.Tensor, "b"],  # speaker
        f0: Float[torch.Tensor, "b n"],
        rms: Float[torch.Tensor, "b n"],
        cvec: Float[torch.Tensor, "b n d2"],
        time: Float[torch.Tensor, "b"],  # time step
        drop_speaker: Union[bool, Bool[torch.Tensor, "b"]] = False,
        mask: Bool[torch.Tensor, "b n"] | None = None,
        skip_layers: Union[int, List[int], None] = None,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = repeat(time, ' -> b', b = batch)
        
        if isinstance(drop_speaker, bool):
            drop_speaker = torch.full((batch,), drop_speaker, dtype=torch.bool, device=x.device)

        spk_embeds = self.spk_embed(spk)
        null_spk_embeds = self.null_spk_embed(torch.zeros_like(spk, dtype=torch.long))
        spk_embeds = torch.where(drop_speaker.unsqueeze(-1), null_spk_embeds, spk_embeds)

        t = self.tembed(time)
        t = t + spk_embeds

        cond_embed = self.cond_embed(f0, rms, cvec)
        x = self.input_embed(x, cond_embed)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if skip_layers is not None:
            if isinstance(skip_layers, int):
                skip_layers = [skip_layers]

        for i, block in enumerate(self.transformer_blocks):
            if skip_layers is not None and i in skip_layers:
                continue
            x = block(x, t, mask = mask, rope = rope)

        x = self.norm_out(x, t)
        output = self.output(x)

        return output
    

    def apply_lora(self, rank, alpha):
        for n, p in self.named_parameters():
            p.requires_grad = False
        self.spk_embed.weight.requires_grad = True
        # Apply LoRA to k_proj and v_proj in each attention block
        for block in self.transformer_blocks:
            block.attn.k_proj = LoRALinear(block.attn.k_proj, rank, alpha)
            block.attn.v_proj = LoRALinear(block.attn.v_proj, rank, alpha)


    def merge_lora(self):
        # Iterate over each transformer block in the DiT backbone
        for block in self.transformer_blocks:
            # Merge for k_proj if it is a LoRALinear instance
            if isinstance(block.attn.k_proj, LoRALinear):
                with torch.no_grad():
                    # Compute delta update: B @ A^T
                    delta = block.attn.k_proj.B @ block.attn.k_proj.A.T
                    # The underlying linear layer has weight of shape (out_features, in_features)
                    # and its forward computes x * weight.T
                    # Note: delta.T equals A @ B^T, so merging works correctly:
                    block.attn.k_proj.linear.weight.add_(delta)
                # Replace the LoRALinear module with the merged linear layer
                block.attn.k_proj = block.attn.k_proj.linear

            # Merge for v_proj in the same way
            if isinstance(block.attn.v_proj, LoRALinear):
                with torch.no_grad():
                    delta = block.attn.v_proj.B @ block.attn.v_proj.A.T
                    block.attn.v_proj.linear.weight.add_(delta)
                block.attn.v_proj = block.attn.v_proj.linear


    def freeze_adaln_and_tembed(self):
        for p in self.tembed.parameters():
            p.requires_grad = False
        for p in self.norm_out.parameters():
            p.requires_grad = False
        for block in self.transformer_blocks:
            for p in block.attn_norm.parameters():
                p.requires_grad = False
