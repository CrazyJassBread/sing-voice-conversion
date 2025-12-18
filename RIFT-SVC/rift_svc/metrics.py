import torch


def psnr(estimated, target, max_val=None):
    """Calculate Peak Signal-to-Noise Ratio (PSNR)
    Args:
        estimated (torch.Tensor): Estimated mel spectrogram [B, len, n_mel]
        target (torch.Tensor): Target mel spectrogram [B, len, n_mel]
        max_val (float): Maximum value of the signal. If None, uses max of target
    Returns:
        torch.Tensor: PSNR value in dB [B]
    """
    if max_val is None:
        # Use the maximum absolute value between both tensors
        max_val = max(torch.abs(target).max(), torch.abs(estimated).max())
    
    # Ensure max_val is not zero
    max_val = max(max_val, torch.finfo(target.dtype).eps)
    
    mse = torch.mean((estimated - target) ** 2, dim=(1, 2))
    # Add eps to avoid log of zero
    eps = torch.finfo(target.dtype).eps
    psnr = 20 * torch.log10(max_val + eps) - 10 * torch.log10(mse + eps)
    return psnr

def si_snr(estimated, target, eps=1e-8):
    """Calculate Scale-Invariant Signal-to-Noise Ratio (SI-SNR)
    Args:
        estimated (torch.Tensor): Estimated mel spectrogram [B, len, n_mel]
        target (torch.Tensor): Target mel spectrogram [B, len, n_mel]
        eps (float): Small value to avoid division by zero
    Returns:
        torch.Tensor: SI-SNR value in dB [B]
    """
    # Flatten the mel dimension
    estimated = estimated.reshape(estimated.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    
    # Zero-mean normalization
    estimated = estimated - torch.mean(estimated, dim=1, keepdim=True)
    target = target - torch.mean(target, dim=1, keepdim=True)
    
    # SI-SNR
    alpha = torch.sum(estimated * target, dim=1, keepdim=True) / (
        torch.sum(target ** 2, dim=1, keepdim=True) + eps)
    target_scaled = alpha * target
    
    si_snr = 10 * torch.log10(
        torch.sum(target_scaled ** 2, dim=1) /
        (torch.sum((estimated - target_scaled) ** 2, dim=1) + eps) + eps
    )
    return si_snr

def mcd(estimated, target):
    """Calculate Mel-Cepstral Distortion (MCD)
    Args:
        estimated (torch.Tensor): Estimated mel spectrogram [B, len, n_mel]
        target (torch.Tensor): Target mel spectrogram [B, len, n_mel]
    Returns:
        torch.Tensor: MCD value [B], averaged over time steps
    """
    # Convert to log scale
    estimated = torch.log10(torch.clamp(estimated, min=1e-8))
    target = torch.log10(torch.clamp(target, min=1e-8))
    
    # Calculate MCD
    diff = estimated - target
    mcd = torch.sqrt(2 * torch.sum(diff ** 2, dim=2))  # [B, len]
    # Average over time dimension
    mcd = mcd.mean(dim=1)  # [B]
    return mcd