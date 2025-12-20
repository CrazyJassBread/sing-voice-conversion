"""
Residual Velocity Adapter (RVA)

This module implements a Residual Velocity Adapter that runs in parallel to the main
DiT backbone to directly manipulate the velocity field for rectified flow.

Formulation:
v_total(Z_t, t, c) = v_base(Z_t, t, c) + α(t) · Δv(Z_t, t, c_target)

Where:
- v_base: Velocity predicted by frozen base DiT
- Δv: Residual velocity correction from RVA
- α(t): Time-dependent guidance scale (higher near t=1 for fine details)
- c_target: Target speaker embedding

The RVA allows the base model to handle melody/structure while taking over
for fine texture generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timestep."""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        self.register_buffer('freqs', freqs)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep tensor [B] or [B, 1]
        Returns:
            Embedding tensor [B, dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        args = t.float() * self.freqs.unsqueeze(0) * 1000
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        return embedding


class TimeDependentGuidanceScale(nn.Module):
    """
    Time-dependent guidance scale α(t) for the RVA.
    
    Design principle: Since coarse structure is determined early (t→0) and
    fine detail later (t→1), α(t) is higher near t=1 to allow the RVA
    to handle fine texture generation.
    
    Options:
    - 'linear': α(t) = α_min + (α_max - α_min) * t
    - 'cosine': α(t) = α_min + (α_max - α_min) * (1 - cos(πt/2))
    - 'learned': Learnable MLP-based schedule
    """
    
    def __init__(
        self,
        schedule_type: str = 'cosine',
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.schedule_type = schedule_type
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        if schedule_type == 'learned':
            self.mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute guidance scale for timestep t.
        
        Args:
            t: Timestep tensor [B] or [B, 1], values in [0, 1]
        Returns:
            Scale tensor [B, 1] or [B]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        if self.schedule_type == 'linear':
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * t
        
        elif self.schedule_type == 'cosine':
            # Cosine schedule: slower increase at start, faster at end
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (1 - torch.cos(math.pi * t / 2))
        
        elif self.schedule_type == 'quadratic':
            # Quadratic schedule: focus on later timesteps
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (t ** 2)
        
        elif self.schedule_type == 'learned':
            alpha = self.mlp(t) * (self.alpha_max - self.alpha_min) + self.alpha_min
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return alpha.squeeze(-1) if alpha.dim() > 1 else alpha


class ResidualBlock(nn.Module):
    """Residual block with optional time conditioning."""
    
    def __init__(
        self,
        channels: int,
        time_emb_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        
        self.dropout = nn.Dropout(dropout)
        
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, channels * 2),
            )
        else:
            self.time_mlp = None
    
    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input [B, C, T]
            t_emb: Time embedding [B, time_emb_dim]
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        if self.time_mlp is not None and t_emb is not None:
            t_proj = self.time_mlp(t_emb)[:, :, None]  # [B, 2C, 1]
            scale, shift = t_proj.chunk(2, dim=1)
            h = h * (1 + scale) + shift
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return x + h


class DownBlock(nn.Module):
    """Downsampling block for U-Net style RVA."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: Optional[int] = None,
        num_layers: int = 2,
        downsample: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                ResidualBlock(
                    in_channels if i == 0 else out_channels,
                    time_emb_dim,
                    dropout,
                )
            )
            if i == 0 and in_channels != out_channels:
                self.blocks.append(nn.Conv1d(in_channels, out_channels, 1))
        
        self.downsample = nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1) if downsample else None
    
    def forward(
        self,
        x: torch.Tensor,
        t_emb: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (downsampled output, skip connection)
        """
        for block in self.blocks:
            if isinstance(block, ResidualBlock):
                x = block(x, t_emb)
            else:
                x = block(x)
        
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block for U-Net style RVA."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        time_emb_dim: Optional[int] = None,
        num_layers: int = 2,
        upsample: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.upsample = nn.ConvTranspose1d(in_channels, in_channels, 4, stride=2, padding=1) if upsample else None
        
        self.skip_conv = nn.Conv1d(in_channels + skip_channels, out_channels, 1)
        
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                ResidualBlock(out_channels, time_emb_dim, dropout)
            )
    
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        t_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='nearest')
        
        x = torch.cat([x, skip], dim=1)
        x = self.skip_conv(x)
        
        for block in self.blocks:
            x = block(x, t_emb)
        
        return x


class ResidualVelocityAdapter(nn.Module):
    """
    Residual Velocity Adapter (RVA) - A lightweight module that predicts
    velocity corrections Δv to steer the rectified flow trajectory.
    
    Architecture: Small U-Net style network (~5M parameters)
    
    Args:
        in_channels: Input mel/latent channels (e.g., 128)
        hidden_channels: Hidden layer channels (e.g., 256)
        out_channels: Output channels (same as in_channels)
        time_emb_dim: Timestep embedding dimension
        style_dim: Target speaker embedding dimension
        cond_dim: Condition (content) embedding dimension
        num_layers: Number of residual blocks per stage
        guidance_schedule: Type of time-dependent guidance ('linear', 'cosine', 'learned')
        alpha_min: Minimum guidance scale
        alpha_max: Maximum guidance scale
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 256,
        out_channels: Optional[int] = None,
        time_emb_dim: int = 256,
        style_dim: int = 192,
        cond_dim: int = 768,
        num_layers: int = 2,
        guidance_schedule: str = 'cosine',
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        dropout: float = 0.1,
        use_unet: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels or in_channels
        self.use_unet = use_unet
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Style (target speaker) embedding projection
        self.style_proj = nn.Sequential(
            nn.Linear(style_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Condition projection (optional - for content features)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_channels),
            nn.SiLU(),
        ) if cond_dim > 0 else None
        
        # Input projection
        input_dim = in_channels + (hidden_channels if cond_dim > 0 else 0)
        self.input_proj = nn.Conv1d(input_dim, hidden_channels, 3, padding=1)
        
        # Time-dependent guidance scale
        self.guidance_scale = TimeDependentGuidanceScale(
            schedule_type=guidance_schedule,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
        
        if use_unet:
            # U-Net style architecture
            self.down1 = DownBlock(hidden_channels, hidden_channels, time_emb_dim, num_layers, downsample=True, dropout=dropout)
            self.down2 = DownBlock(hidden_channels, hidden_channels * 2, time_emb_dim, num_layers, downsample=True, dropout=dropout)
            
            self.mid = ResidualBlock(hidden_channels * 2, time_emb_dim, dropout)
            
            self.up2 = UpBlock(hidden_channels * 2, hidden_channels, hidden_channels * 2, time_emb_dim, num_layers, upsample=True, dropout=dropout)
            self.up1 = UpBlock(hidden_channels, hidden_channels, hidden_channels, time_emb_dim, num_layers, upsample=True, dropout=dropout)
        else:
            # Simple MLP/ResNet style
            self.blocks = nn.ModuleList([
                ResidualBlock(hidden_channels, time_emb_dim, dropout)
                for _ in range(4)
            ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, self.out_channels, 3, padding=1),
        )
        
        # Initialize output to zero for stable training
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)
    
    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        style: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        return_unscaled: bool = False,
    ) -> torch.Tensor:
        """
        Predict residual velocity correction.
        
        Args:
            z_t: Noisy latent [B, C, T] (mel channels, time frames)
            t: Timestep [B] in [0, 1]
            style: Target speaker embedding [B, style_dim]
            cond: Optional content condition [B, T, cond_dim]
            return_unscaled: If True, return Δv without guidance scaling
            
        Returns:
            Scaled residual velocity: α(t) · Δv [B, C, T]
        """
        B, C, T = z_t.shape
        
        # Time and style embeddings
        t_emb = self.time_embed(t)  # [B, time_emb_dim]
        style_emb = self.style_proj(style)  # [B, time_emb_dim]
        
        # Combine time and style
        combined_emb = t_emb + style_emb  # [B, time_emb_dim]
        
        # Process condition if provided
        if cond is not None and self.cond_proj is not None:
            # cond: [B, T, cond_dim] -> [B, hidden_channels, T]
            cond_feat = self.cond_proj(cond).transpose(1, 2)
            x = torch.cat([z_t, cond_feat], dim=1)
        else:
            x = z_t
        
        # Input projection
        x = self.input_proj(x)  # [B, hidden_channels, T]
        
        if self.use_unet:
            # U-Net forward
            x, skip1 = self.down1(x, combined_emb)
            x, skip2 = self.down2(x, combined_emb)
            
            x = self.mid(x, combined_emb)
            
            x = self.up2(x, skip2, combined_emb)
            x = self.up1(x, skip1, combined_emb)
        else:
            # Simple ResNet forward
            for block in self.blocks:
                x = block(x, combined_emb)
        
        # Output projection
        delta_v = self.output_proj(x)  # [B, out_channels, T]
        
        if return_unscaled:
            return delta_v
        
        # Apply time-dependent guidance scale
        alpha = self.guidance_scale(t)  # [B]
        scaled_delta_v = alpha[:, None, None] * delta_v
        
        return scaled_delta_v
    
    def get_total_velocity(
        self,
        v_base: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        style: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute total velocity: v_total = v_base + α(t) · Δv
        
        Args:
            v_base: Base velocity from frozen DiT [B, C, T]
            z_t: Noisy latent [B, C, T]
            t: Timestep [B]
            style: Target speaker embedding [B, style_dim]
            cond: Optional content condition [B, T, cond_dim]
            
        Returns:
            Total velocity [B, C, T]
        """
        delta_v = self.forward(z_t, t, style, cond)
        return v_base + delta_v


class RVAWithDiT(nn.Module):
    """
    Wrapper that combines frozen DiT with trainable RVA.
    
    This module:
    1. Computes v_base from frozen DiT
    2. Computes Δv from RVA
    3. Returns v_total = v_base + α(t) · Δv
    """
    
    def __init__(
        self,
        dit_estimator: nn.Module,
        rva_config: dict,
        freeze_dit: bool = True,
    ):
        super().__init__()
        
        self.dit = dit_estimator
        self.rva = ResidualVelocityAdapter(**rva_config)
        
        if freeze_dit:
            for param in self.dit.parameters():
                param.requires_grad = False
            self.dit.eval()
    
    def forward(
        self,
        x: torch.Tensor,
        prompt_x: torch.Tensor,
        x_lens: torch.Tensor,
        t: torch.Tensor,
        style: torch.Tensor,
        cond: torch.Tensor,
        use_rva: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with optional RVA correction.
        
        Args:
            x: Noisy latent [B, C, T]
            prompt_x: Prompt mel [B, C, T]
            x_lens: Length tensor
            t: Timestep [B]
            style: Speaker embedding [B, style_dim]
            cond: Content condition [B, T, cond_dim]
            use_rva: Whether to apply RVA correction
        """
        # Get base velocity from DiT
        with torch.no_grad():
            v_base = self.dit(x, prompt_x, x_lens, t, style, cond, **kwargs)
        
        if not use_rva:
            return v_base
        
        # Get RVA correction
        # Note: RVA uses transposed input format [B, C, T]
        delta_v = self.rva(x, t, style, cond.transpose(1, 2) if cond is not None else None)
        
        return v_base + delta_v
    
    def get_trainable_params(self) -> list:
        """Get only RVA parameters for training."""
        return list(self.rva.parameters())


def create_rva(
    in_channels: int = 128,
    hidden_channels: int = 256,
    style_dim: int = 192,
    cond_dim: int = 768,
    guidance_schedule: str = 'cosine',
    alpha_max: float = 0.5,
    use_unet: bool = True,
) -> ResidualVelocityAdapter:
    """
    Factory function to create RVA with default settings.
    
    Approximate parameter count with defaults: ~5M parameters
    """
    return ResidualVelocityAdapter(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=in_channels,
        time_emb_dim=256,
        style_dim=style_dim,
        cond_dim=cond_dim,
        num_layers=2,
        guidance_schedule=guidance_schedule,
        alpha_min=0.0,
        alpha_max=alpha_max,
        dropout=0.1,
        use_unet=use_unet,
    )


def count_rva_params(rva: ResidualVelocityAdapter) -> int:
    """Count total trainable parameters in RVA."""
    return sum(p.numel() for p in rva.parameters() if p.requires_grad)
