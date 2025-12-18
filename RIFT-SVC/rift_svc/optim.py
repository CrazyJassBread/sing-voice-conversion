from schedulefree import AdamWScheduleFree
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
import math


def get_optimizer(
        optimizer_type, model, lr, betas, weight_decay, warmup_steps, 
        lora_training=False, **kwargs):
    from collections import defaultdict
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    if not lora_training:
        specp_decay_params = defaultdict(list)
        specp_decay_lr = {}
        decay_params = []
        nodecay_params = []
        for n, p in param_dict.items():
            if p.dim() >= 2:
                if n.endswith('out.weight') or n.endswith('proj.weight'):
                    fan_out, fan_in = p.shape[-2:]
                    fan_ratio = fan_out / fan_in
                    specp_decay_params[f"specp_decay_{fan_ratio:.2f}"].append(p)
                    specp_decay_lr[f"specp_decay_{fan_ratio:.2f}"] = lr * fan_ratio
                else:
                    decay_params.append(p)
            else:
                nodecay_params.append(p)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': lr},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': lr}
        ] + [
            {'params': params, 'weight_decay': weight_decay, 'lr': specp_decay_lr[group_name]}
            for group_name, params in specp_decay_params.items()
        ]
    else:
        lora_a_or_spk_embed_params = []
        lora_b_params = []
        for n, p in param_dict.items():
            if n.endswith('.A.weight') or n.endswith('.spk_embed.weight'):
                lora_a_or_spk_embed_params.append(p)
            elif n.endswith('.B.weight'):
                lora_b_params.append(p)
        dim = model.transformer.dim
        rank = model.transformer.transformer_blocks[0].attn.k_proj.rank
        optim_groups = [
            {'params': lora_a_or_spk_embed_params, 'weight_decay': weight_decay, 'lr': lr},
            {'params': lora_b_params, 'weight_decay': weight_decay, 'lr': lr*math.sqrt(dim/rank)}
        ]
    if optimizer_type == 'adamwsf':
        optimizer = AdamWScheduleFree(optim_groups, betas=betas, warmup_steps=warmup_steps)
        return optimizer, None
    elif optimizer_type == 'adamw':
        optimizer = AdamW(optim_groups, betas=betas, weight_decay=weight_decay)
        max_steps = kwargs['max_steps']
        min_lr = kwargs.get('min_lr', 0.0)
        lr_scheduler = LinearWarmupDecayLR(optimizer, warmup_steps, max_steps, min_lr=min_lr)
        return optimizer, lr_scheduler
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")


class LinearWarmupDecayLR(_LRScheduler):
    """
    Linear learning rate scheduler with warmup and minimum lr.
    
    During warmup, the LR increases linearly from 0 to the base LR.
    After warmup, the LR decays linearly from the base LR down to min_lr.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of steps to linearly increase LR.
        total_steps (int): Total number of steps for training (warmup + decay).
        min_lr (float): Minimum learning rate after decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1):
        if total_steps <= warmup_steps:
            raise ValueError(
                "Total steps must be larger than warmup_steps for decay to happen."
            )
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super(LinearWarmupDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate using linear warmup and then linear decay."""
        # Note: self.last_epoch is incremented by the base _LRScheduler.step() before calling get_lr().
        if self.last_epoch < self.warmup_steps:
            # Warmup phase: increase linearly from 0 (or a small value) to base_lr.
            return [
                base_lr * float(self.last_epoch + 1) / float(self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            # Decay phase: decrease linearly from base_lr to min_lr.
            progress = float(self.last_epoch - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
            return [
                max(base_lr * (1.0 - progress) + self.min_lr * progress, self.min_lr)
                for base_lr in self.base_lrs
            ]