"""
Adapters Module - Integration utilities for SFA-LoRA and RVA

This module provides high-level integration utilities for applying
Spectral-Feature-Aware LoRA and Residual Velocity Adapter to the
YingMusic-SVC DiT model.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple, Any
from munch import Munch

from modules.sfa_lora import (
    SFALoRALinear,
    SFALoRAAttention,
    FrequencyGate,
    inject_sfa_lora,
    get_lora_params,
    freeze_base_model,
    count_lora_params,
)
from modules.rva import (
    ResidualVelocityAdapter,
    RVAWithDiT,
    create_rva,
    count_rva_params,
)


class AdaptedCFM(nn.Module):
    """
    CFM (Conditional Flow Matching) with SFA-LoRA and RVA adaptation.
    
    This wrapper adds:
    1. SFA-LoRA injection into the DiT attention layers
    2. RVA for residual velocity correction
    
    Training modes:
    - 'lora_only': Train only SFA-LoRA parameters
    - 'rva_only': Train only RVA parameters  
    - 'both': Train both SFA-LoRA and RVA
    - 'full': Train all parameters (fine-tuning)
    """
    
    def __init__(
        self,
        base_cfm: nn.Module,
        lora_config: Optional[dict] = None,
        rva_config: Optional[dict] = None,
        training_mode: str = 'both',
        freeze_base: bool = True,
    ):
        super().__init__()
        
        self.base_cfm = base_cfm
        self.training_mode = training_mode
        self.freeze_base = freeze_base
        
        # Store configs
        self.lora_config = lora_config or {}
        self.rva_config = rva_config or {}
        
        # Initialize adapters
        self.lora_modules = {}
        self.rva = None
        
        self._use_lora = lora_config is not None and len(lora_config) > 0
        self._use_rva = rva_config is not None and len(rva_config) > 0
        
        if self._use_lora:
            self._inject_lora()
        
        if self._use_rva:
            self._init_rva()
        
        if freeze_base:
            self._freeze_base_model()
        
        self._setup_training_mode()
    
    def _inject_lora(self):
        """Inject SFA-LoRA into the DiT estimator."""
        estimator = self.base_cfm.estimator
        
        # Default LoRA config
        rank = self.lora_config.get('rank', 8)
        alpha = self.lora_config.get('alpha', 16.0)
        num_freq_bins = self.lora_config.get('num_freq_bins', 128)
        target_modules = self.lora_config.get('target_modules', ('wqkv', 'wo'))
        dropout = self.lora_config.get('dropout', 0.0)
        sr = self.lora_config.get('sr', 44100)
        
        self.lora_modules = inject_sfa_lora(
            model=estimator,
            rank=rank,
            alpha=alpha,
            num_freq_bins=num_freq_bins,
            target_modules=target_modules,
            dropout=dropout,
            sr=sr,
        )
        
        print(f"Injected SFA-LoRA into {len(self.lora_modules)} modules")
        lora_params, total_params = count_lora_params(estimator)
        print(f"LoRA parameters: {lora_params:,} / {total_params:,} total ({100*lora_params/total_params:.2f}%)")
    
    def _init_rva(self):
        """Initialize the Residual Velocity Adapter."""
        in_channels = self.rva_config.get('in_channels', 128)
        hidden_channels = self.rva_config.get('hidden_channels', 256)
        style_dim = self.rva_config.get('style_dim', 192)
        cond_dim = self.rva_config.get('cond_dim', 768)
        guidance_schedule = self.rva_config.get('guidance_schedule', 'cosine')
        alpha_min = self.rva_config.get('alpha_min', 0.0)
        alpha_max = self.rva_config.get('alpha_max', 0.5)
        use_unet = self.rva_config.get('use_unet', True)
        
        self.rva = ResidualVelocityAdapter(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            style_dim=style_dim,
            cond_dim=cond_dim,
            guidance_schedule=guidance_schedule,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            use_unet=use_unet,
        )
        
        rva_params = count_rva_params(self.rva)
        print(f"RVA parameters: {rva_params:,}")
    
    def _freeze_base_model(self):
        """Freeze base model parameters."""
        for name, param in self.base_cfm.named_parameters():
            # Don't freeze LoRA parameters
            if 'lora_' in name.lower() or 'freq_gate' in name.lower():
                continue
            param.requires_grad = False
    
    def _setup_training_mode(self):
        """Setup which parameters are trainable based on training mode."""
        if self.training_mode == 'lora_only':
            # Only LoRA parameters
            if self.rva is not None:
                for param in self.rva.parameters():
                    param.requires_grad = False
        
        elif self.training_mode == 'rva_only':
            # Only RVA parameters
            for name, param in self.base_cfm.named_parameters():
                if 'lora_' in name.lower() or 'freq_gate' in name.lower():
                    param.requires_grad = False
        
        elif self.training_mode == 'full':
            # Unfreeze everything
            for param in self.base_cfm.parameters():
                param.requires_grad = True
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters based on training mode."""
        params = []
        
        if self.training_mode in ['lora_only', 'both', 'full']:
            params.extend(get_lora_params(self.base_cfm, include_gate=True))
        
        if self.training_mode in ['rva_only', 'both'] and self.rva is not None:
            params.extend(list(self.rva.parameters()))
        
        if self.training_mode == 'full':
            # Add all base model params
            for param in self.base_cfm.parameters():
                if param.requires_grad and param not in params:
                    params.append(param)
        
        return params
    
    def forward(
        self,
        x1: torch.Tensor,
        x_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
        mu: torch.Tensor,
        style: torch.Tensor,
        use_rva: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional RVA correction.
        
        Args:
            x1: Target mel [B, C, T]
            x_lens: Lengths [B]
            prompt_lens: Prompt lengths [B]
            mu: Content condition [B, T, D]
            style: Style embedding [B, style_dim]
            use_rva: Whether to apply RVA correction
            
        Returns:
            loss, predicted output
        """
        b, _, t = x1.shape
        
        # Random timestep
        t_diff = torch.rand([b, 1, 1], device=mu.device, dtype=x1.dtype)
        
        # Sample noise
        z = torch.randn_like(x1)
        
        # Interpolate: y = (1 - (1-σ_min)t)z + tx1
        sigma_min = self.base_cfm.sigma_min
        y = (1 - (1 - sigma_min) * t_diff) * z + t_diff * x1
        u = x1 - (1 - sigma_min) * z  # Target velocity
        
        # Setup prompt
        prompt = torch.zeros_like(x1)
        for bib in range(b):
            prompt[bib, :, :prompt_lens[bib]] = x1[bib, :, :prompt_lens[bib]]
            y[bib, :, :prompt_lens[bib]] = 0
            if self.base_cfm.zero_prompt_speech_token:
                mu[bib, :, :prompt_lens[bib]] = 0
        
        # Get base velocity from DiT (with LoRA if injected)
        t_squeezed = t_diff.squeeze(1).squeeze(1)
        v_base = self.base_cfm.estimator(
            y, prompt, x_lens, t_squeezed, style, mu, prompt_lens, **kwargs
        )
        
        # Add RVA correction if enabled
        if use_rva and self.rva is not None:
            # RVA expects [B, C, T] for z_t, and [B, T, D] for cond
            delta_v = self.rva(y, t_squeezed, style, mu)
            estimator_out = v_base + delta_v
        else:
            estimator_out = v_base
        
        # Compute loss
        loss = 0
        criterion = self.base_cfm.criterion
        for bib in range(b):
            loss += criterion(
                estimator_out[bib, :, prompt_lens[bib]:x_lens[bib]],
                u[bib, :, prompt_lens[bib]:x_lens[bib]]
            )
        loss /= b
        
        return loss, estimator_out + (1 - sigma_min) * z
    
    def inference(
        self,
        mu: torch.Tensor,
        x_lens: torch.Tensor,
        prompt: torch.Tensor,
        style: torch.Tensor,
        f0: Optional[torch.Tensor] = None,
        n_timesteps: int = 32,
        temperature: float = 1.0,
        inference_cfg_rate: float = 0.5,
        use_rva: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Inference with RVA-enhanced velocity field.
        
        Args:
            mu: Content condition [B, T, D]
            x_lens: Lengths [B]
            prompt: Prompt mel [B, C, P]
            style: Style embedding [B, style_dim]
            n_timesteps: Number of ODE steps
            temperature: Noise temperature
            inference_cfg_rate: CFG rate
            use_rva: Whether to use RVA correction
            
        Returns:
            Generated mel spectrogram [B, C, T]
        """
        B, T = mu.size(0), mu.size(1)
        device = mu.device
        
        # Initial noise
        z = torch.randn([B, self.base_cfm.in_channels, T], device=device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=device)
        
        return self._solve_euler_with_rva(
            z, x_lens, prompt, mu, style, t_span,
            inference_cfg_rate, use_rva, **kwargs
        )
    
    def _solve_euler_with_rva(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        prompt: torch.Tensor,
        mu: torch.Tensor,
        style: torch.Tensor,
        t_span: torch.Tensor,
        inference_cfg_rate: float = 0.5,
        use_rva: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """Euler ODE solver with RVA correction."""
        t = t_span[0]
        
        # Setup prompt region
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        
        if self.base_cfm.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0
        
        for step in range(1, len(t_span)):
            dt = t_span[step] - t_span[step - 1]
            
            # Get velocity
            if inference_cfg_rate > 0:
                # CFG: Stack original and null inputs
                stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
                stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
                stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
                stacked_x = torch.cat([x, x], dim=0)
                stacked_t = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)
                
                # Base velocity
                stacked_v_base = self.base_cfm.estimator(
                    stacked_x, stacked_prompt_x, x_lens, stacked_t, stacked_style, stacked_mu, **kwargs
                )
                
                # Apply CFG
                v_base, cfg_v_base = stacked_v_base.chunk(2, dim=0)
                v_base = (1.0 + inference_cfg_rate) * v_base - inference_cfg_rate * cfg_v_base
                
                # RVA correction (only on conditioned branch)
                if use_rva and self.rva is not None:
                    delta_v = self.rva(x, t.unsqueeze(0).expand(x.size(0)), style, mu)
                    dphi_dt = v_base + delta_v
                else:
                    dphi_dt = v_base
            else:
                v_base = self.base_cfm.estimator(
                    x, prompt_x, x_lens, t.unsqueeze(0), style, mu, **kwargs
                )
                
                if use_rva and self.rva is not None:
                    delta_v = self.rva(x, t.unsqueeze(0).expand(x.size(0)), style, mu)
                    dphi_dt = v_base + delta_v
                else:
                    dphi_dt = v_base
            
            # Euler step
            x = x + dt * dphi_dt
            t = t + dt
            
            # Maintain prompt constraint
            if step < len(t_span) - 1:
                x[:, :, :prompt_len] = 0
        
        return x
    
    def save_adapters(self, path: str):
        """Save adapter weights (LoRA + RVA)."""
        state = {
            'lora_config': self.lora_config,
            'rva_config': self.rva_config,
            'training_mode': self.training_mode,
        }
        
        # Save LoRA weights
        if self._use_lora:
            lora_state = {}
            for name, param in self.base_cfm.named_parameters():
                if 'lora_' in name.lower() or 'freq_gate' in name.lower():
                    lora_state[name] = param.data
            state['lora_state'] = lora_state
        
        # Save RVA weights
        if self._use_rva and self.rva is not None:
            state['rva_state'] = self.rva.state_dict()
        
        torch.save(state, path)
        print(f"Saved adapters to {path}")
    
    def load_adapters(self, path: str, strict: bool = True):
        """Load adapter weights."""
        state = torch.load(path, map_location='cpu')
        
        # Load LoRA weights
        if 'lora_state' in state:
            model_state = dict(self.base_cfm.named_parameters())
            for name, param in state['lora_state'].items():
                if name in model_state:
                    model_state[name].data.copy_(param)
            print("Loaded LoRA weights")
        
        # Load RVA weights
        if 'rva_state' in state and self.rva is not None:
            self.rva.load_state_dict(state['rva_state'], strict=strict)
            print("Loaded RVA weights")


def create_adapted_cfm(
    base_cfm: nn.Module,
    use_lora: bool = True,
    use_rva: bool = True,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    rva_hidden: int = 256,
    rva_alpha_max: float = 0.5,
    training_mode: str = 'both',
    **kwargs
) -> AdaptedCFM:
    """
    Factory function to create an adapted CFM with SFA-LoRA and RVA.
    
    Args:
        base_cfm: Base CFM model
        use_lora: Whether to use SFA-LoRA
        use_rva: Whether to use RVA
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        rva_hidden: RVA hidden channels
        rva_alpha_max: RVA max guidance scale
        training_mode: 'lora_only', 'rva_only', 'both', or 'full'
        
    Returns:
        AdaptedCFM instance
    """
    lora_config = None
    rva_config = None
    
    if use_lora:
        lora_config = {
            'rank': lora_rank,
            'alpha': lora_alpha,
            'num_freq_bins': kwargs.get('num_freq_bins', 128),
            'target_modules': kwargs.get('target_modules', ('wqkv', 'wo')),
            'dropout': kwargs.get('lora_dropout', 0.0),
            'sr': kwargs.get('sr', 44100),
        }
    
    if use_rva:
        rva_config = {
            'in_channels': kwargs.get('in_channels', 128),
            'hidden_channels': rva_hidden,
            'style_dim': kwargs.get('style_dim', 192),
            'cond_dim': kwargs.get('cond_dim', 768),
            'guidance_schedule': kwargs.get('guidance_schedule', 'cosine'),
            'alpha_min': kwargs.get('rva_alpha_min', 0.0),
            'alpha_max': rva_alpha_max,
            'use_unet': kwargs.get('rva_use_unet', True),
        }
    
    return AdaptedCFM(
        base_cfm=base_cfm,
        lora_config=lora_config,
        rva_config=rva_config,
        training_mode=training_mode,
        freeze_base=kwargs.get('freeze_base', True),
    )


def count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
