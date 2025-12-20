"""
Spectral-Feature-Aware LoRA (SFA-LoRA)

This module implements a modified Low-Rank Adaptation technique that incorporates
a frequency gating mechanism to scale LoRA contributions based on frequency bins.

Standard LoRA: h = Wx + BAx
SFA-LoRA:      h(t,f) = Wx(t,f) + Γ(f) ⊙ (BAx(t,f))

Where Γ(f) is a learnable frequency gate that allows the model to learn different
adaptation strengths for different frequency bands:
- Formant bands (1-3kHz): Higher adaptation
- Breath bands (>5kHz): Higher adaptation  
- Fundamental bands (<500Hz): Lower adaptation (shared physics)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class FrequencyGate(nn.Module):
    """
    Learnable frequency gating mechanism Γ(f) that scales LoRA contributions
    based on frequency bins.
    
    The gate is initialized with a prior that encourages:
    - Low adaptation for fundamental frequencies (<500Hz)
    - High adaptation for formant frequencies (1-3kHz)
    - High adaptation for breath/air frequencies (>5kHz)
    
    Args:
        num_freq_bins: Number of frequency bins (e.g., 128 for mel-spectrogram)
        sr: Sample rate in Hz (default 44100)
        learnable: Whether the gate is learnable (default True)
        init_type: Initialization type ('uniform', 'frequency_prior', 'ones')
    """
    
    def __init__(
        self,
        num_freq_bins: int = 128,
        sr: int = 44100,
        learnable: bool = True,
        init_type: str = 'frequency_prior',
        fmin: float = 0.0,
        fmax: float = None,
    ):
        super().__init__()
        self.num_freq_bins = num_freq_bins
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sr / 2
        
        # Initialize the gate values
        gate_init = self._init_gate(init_type)
        
        if learnable:
            self.gate = nn.Parameter(gate_init)
        else:
            self.register_buffer('gate', gate_init)
    
    def _init_gate(self, init_type: str) -> torch.Tensor:
        """Initialize gate values based on initialization type."""
        if init_type == 'uniform':
            return torch.ones(self.num_freq_bins) * 0.5
        
        elif init_type == 'ones':
            return torch.ones(self.num_freq_bins)
        
        elif init_type == 'frequency_prior':
            # Create frequency-aware initialization
            # Mel-scale frequency centers for each bin
            mel_min = 2595 * math.log10(1 + self.fmin / 700)
            mel_max = 2595 * math.log10(1 + self.fmax / 700)
            mels = torch.linspace(mel_min, mel_max, self.num_freq_bins)
            freqs = 700 * (10 ** (mels / 2595) - 1)  # Hz
            
            # Initialize gate based on frequency bands
            gate = torch.ones(self.num_freq_bins) * 0.5
            
            for i, freq in enumerate(freqs):
                if freq < 500:  # Fundamental band - low adaptation
                    gate[i] = 0.2
                elif 500 <= freq < 1000:  # Transition band
                    gate[i] = 0.4
                elif 1000 <= freq < 3000:  # Formant band - high adaptation
                    gate[i] = 0.8
                elif 3000 <= freq < 5000:  # Upper formants
                    gate[i] = 0.6
                else:  # Breath/air band (>5kHz) - high adaptation
                    gate[i] = 0.7
            
            return gate
        
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency gating.
        
        Args:
            x: Input tensor, can be:
               - [B, T, D] where D contains frequency info
               - [B, C, T] where C is frequency dimension
               
        Returns:
            Gated tensor of same shape
        """
        # Apply sigmoid to constrain gate values to [0, 1]
        gate = torch.sigmoid(self.gate)
        
        # Handle different tensor shapes
        if x.dim() == 3:
            # Assume frequency is the last dimension or channel dimension
            if x.size(-1) == self.num_freq_bins:
                # [B, T, F] format
                return x * gate.view(1, 1, -1)
            elif x.size(1) == self.num_freq_bins:
                # [B, F, T] format
                return x * gate.view(1, -1, 1)
        
        # Check 2D case [B, D] or [N, D]
        if x.dim() == 2:
            if x.size(-1) == self.num_freq_bins:
                return x * gate.view(1, -1)
        
        # Fallback: if dimensions don't match, use mean gate value
        # This happens when the feature dimension D != num_freq_bins
        # We use the average gate value to scale the features
        return x * gate.mean()


class SFALoRALinear(nn.Module):
    """
    Spectral-Feature-Aware LoRA for Linear layers.
    
    Implements: h(t,f) = Wx(t,f) + α * Γ(f) ⊙ (BAx(t,f))
    
    Args:
        original_layer: The original nn.Linear layer to adapt
        rank: LoRA rank (r)
        alpha: LoRA scaling factor
        num_freq_bins: Number of frequency bins for the gate
        dropout: Dropout rate for LoRA
        use_freq_gate: Whether to use frequency gating
        gate_init_type: Initialization type for frequency gate
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        num_freq_bins: int = 128,
        dropout: float = 0.0,
        use_freq_gate: bool = True,
        gate_init_type: str = 'frequency_prior',
        sr: int = 44100,
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.use_freq_gate = use_freq_gate
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA matrices: B ∈ R^{d×r}, A ∈ R^{r×d}
        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Frequency gate
        if use_freq_gate:
            self.freq_gate = FrequencyGate(
                num_freq_bins=num_freq_bins,
                sr=sr,
                init_type=gate_init_type
            )
        else:
            self.freq_gate = None
    
    def forward(self, x: torch.Tensor, freq_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with SFA-LoRA.
        
        Args:
            x: Input tensor [B, T, D]
            freq_indices: Optional frequency bin indices for each token
            
        Returns:
            Output tensor with LoRA adaptation
        """
        # Original forward
        result = self.original_layer(x)
        
        # LoRA forward: BA @ x
        lora_output = self.lora_B(self.lora_A(self.dropout(x)))
        
        # Apply frequency gating if enabled
        if self.use_freq_gate and self.freq_gate is not None:
            # Get the gate values
            gate = torch.sigmoid(self.freq_gate.gate)  # [num_freq_bins]
            
            if freq_indices is not None:
                # Use provided frequency indices
                # freq_indices: [B, T] -> gate values for each position
                batch_gate = gate[freq_indices]  # [B, T]
                lora_output = lora_output * batch_gate.unsqueeze(-1)
            else:
                # Apply gate based on output dimension structure
                # Assume the output represents frequency-related features
                lora_output = self.freq_gate(lora_output)
        
        # Scale and add
        result = result + self.scaling * lora_output
        
        return result


class SFALoRAAttention(nn.Module):
    """
    Wrapper to add SFA-LoRA to attention layers (Q, K, V, O projections).
    
    Args:
        attention_module: The original attention module
        rank: LoRA rank
        alpha: LoRA scaling factor
        num_freq_bins: Number of frequency bins
        target_modules: Which projections to adapt ('q', 'k', 'v', 'o')
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        rank: int = 8,
        alpha: float = 16.0,
        num_freq_bins: int = 128,
        target_modules: Tuple[str, ...] = ('q', 'v'),
        dropout: float = 0.0,
        sr: int = 44100,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.target_modules = target_modules
        
        total_head_dim = num_heads * head_dim
        
        # Create LoRA layers for target modules
        self.lora_layers = nn.ModuleDict()
        
        if 'q' in target_modules:
            self.lora_layers['q_A'] = nn.Linear(dim, rank, bias=False)
            self.lora_layers['q_B'] = nn.Linear(rank, total_head_dim, bias=False)
            nn.init.kaiming_uniform_(self.lora_layers['q_A'].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_layers['q_B'].weight)
        
        if 'k' in target_modules:
            self.lora_layers['k_A'] = nn.Linear(dim, rank, bias=False)
            self.lora_layers['k_B'] = nn.Linear(rank, total_head_dim, bias=False)
            nn.init.kaiming_uniform_(self.lora_layers['k_A'].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_layers['k_B'].weight)
        
        if 'v' in target_modules:
            self.lora_layers['v_A'] = nn.Linear(dim, rank, bias=False)
            self.lora_layers['v_B'] = nn.Linear(rank, total_head_dim, bias=False)
            nn.init.kaiming_uniform_(self.lora_layers['v_A'].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_layers['v_B'].weight)
        
        if 'o' in target_modules:
            self.lora_layers['o_A'] = nn.Linear(total_head_dim, rank, bias=False)
            self.lora_layers['o_B'] = nn.Linear(rank, dim, bias=False)
            nn.init.kaiming_uniform_(self.lora_layers['o_A'].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_layers['o_B'].weight)
        
        # Frequency gate
        self.freq_gate = FrequencyGate(
            num_freq_bins=num_freq_bins,
            sr=sr,
            init_type='frequency_prior'
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def get_lora_q(self, x: torch.Tensor) -> torch.Tensor:
        """Get LoRA contribution for Q projection."""
        if 'q' not in self.target_modules:
            return torch.zeros_like(x[..., :self.num_heads * self.head_dim])
        lora_out = self.lora_layers['q_B'](self.lora_layers['q_A'](self.dropout(x)))
        return self._apply_freq_gate(lora_out) * self.scaling
    
    def get_lora_k(self, x: torch.Tensor) -> torch.Tensor:
        """Get LoRA contribution for K projection."""
        if 'k' not in self.target_modules:
            return torch.zeros_like(x[..., :self.num_heads * self.head_dim])
        lora_out = self.lora_layers['k_B'](self.lora_layers['k_A'](self.dropout(x)))
        return self._apply_freq_gate(lora_out) * self.scaling
    
    def get_lora_v(self, x: torch.Tensor) -> torch.Tensor:
        """Get LoRA contribution for V projection."""
        if 'v' not in self.target_modules:
            return torch.zeros_like(x[..., :self.num_heads * self.head_dim])
        lora_out = self.lora_layers['v_B'](self.lora_layers['v_A'](self.dropout(x)))
        return self._apply_freq_gate(lora_out) * self.scaling
    
    def get_lora_o(self, x: torch.Tensor) -> torch.Tensor:
        """Get LoRA contribution for output projection."""
        if 'o' not in self.target_modules:
            return torch.zeros_like(x[..., :self.dim])
        lora_out = self.lora_layers['o_B'](self.lora_layers['o_A'](self.dropout(x)))
        return self._apply_freq_gate(lora_out) * self.scaling
    
    def _apply_freq_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply frequency gating to LoRA output."""
        gate = torch.sigmoid(self.freq_gate.gate)  # [num_freq_bins]
        
        # For attention features, we apply a global gate based on 
        # the average frequency contribution
        B, T, D = x.shape
        
        # Reshape to apply per-position gating
        # We use a soft weighting based on learned gate values
        gate_weight = gate.mean()  # Simple average for now
        return x * gate_weight


def inject_sfa_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    num_freq_bins: int = 128,
    target_modules: Tuple[str, ...] = ('wqkv', 'wo'),
    dropout: float = 0.0,
    sr: int = 44100,
) -> Dict[str, nn.Module]:
    """
    Inject SFA-LoRA into a model's linear layers.
    
    Args:
        model: The model to inject LoRA into
        rank: LoRA rank
        alpha: LoRA scaling factor
        num_freq_bins: Number of frequency bins
        target_modules: Names of modules to inject LoRA into
        dropout: Dropout rate
        sr: Sample rate
        
    Returns:
        Dictionary of injected LoRA modules
    """
    lora_modules = {}
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Get parent module and attribute name
                parts = name.rsplit('.', 1)
                if len(parts) == 1:
                    parent = model
                    attr_name = parts[0]
                else:
                    parent = dict(model.named_modules())[parts[0]]
                    attr_name = parts[1]
                
                # Create SFA-LoRA wrapper
                lora_linear = SFALoRALinear(
                    original_layer=module,
                    rank=rank,
                    alpha=alpha,
                    num_freq_bins=num_freq_bins,
                    dropout=dropout,
                    use_freq_gate=True,
                    sr=sr,
                )
                
                # Replace the module
                setattr(parent, attr_name, lora_linear)
                lora_modules[name] = lora_linear
                
                print(f"Injected SFA-LoRA into {name}")
    
    return lora_modules


def get_lora_params(model: nn.Module, include_gate: bool = True) -> list:
    """
    Get all trainable LoRA parameters from a model.
    
    Args:
        model: Model with injected LoRA
        include_gate: Whether to include frequency gate parameters
        
    Returns:
        List of trainable parameters
    """
    params = []
    
    for name, param in model.named_parameters():
        if 'lora_' in name.lower():
            params.append(param)
        elif include_gate and 'freq_gate' in name.lower():
            params.append(param)
    
    return params


def freeze_base_model(model: nn.Module):
    """Freeze all parameters except LoRA parameters."""
    for name, param in model.named_parameters():
        if 'lora_' not in name.lower() and 'freq_gate' not in name.lower():
            param.requires_grad = False
        else:
            param.requires_grad = True


def count_lora_params(model: nn.Module) -> Tuple[int, int]:
    """
    Count LoRA and total parameters.
    
    Returns:
        Tuple of (lora_params, total_params)
    """
    lora_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if 'lora_' in name.lower() or 'freq_gate' in name.lower():
            lora_params += param.numel()
    
    return lora_params, total_params
