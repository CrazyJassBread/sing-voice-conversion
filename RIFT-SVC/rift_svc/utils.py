import io
import os
import random
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Int
from PIL import Image
from pytorch_lightning.callbacks import TQDMProgressBar
import parselmouth as pm
import librosa
import pyworld as pw


def seed_everything(seed: int = 0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# helpers

def exists(v: Any) -> bool:
    return v is not None

def default(v: Any, d: Any) -> Any:
    return v if exists(v) else d


def draw_mel_specs(gt: np.ndarray, gen: np.ndarray, diff: np.ndarray, cache_path: str):
    vmin = min(gt.min(), gen.min())
    vmax = max(gt.max(), gen.max())
    
    # Create figure with space for colorbar
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15), sharex=True, gridspec_kw={'hspace': 0})
    
    # Plot all spectrograms with the same scale
    im1 = ax1.imshow(gt, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    ax1.set_ylabel('GT', fontsize=14)
    ax1.set_xticks([])
    
    im2 = ax2.imshow(gen, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    ax2.set_ylabel('Gen', fontsize=14)
    ax2.set_xticks([])
    
    # Find symmetric limits for difference plot
    diff_abs_max = max(abs(diff.min()), abs(diff.max()))
    
    im3 = ax3.imshow(diff, origin='lower', aspect='auto',
                     cmap='RdBu_r',  # Red-White-Blue colormap (reversed)
                     vmin=-diff_abs_max, vmax=diff_abs_max)
    ax3.set_ylabel('Diff', fontsize=14)
    
    fig.colorbar(im1, ax=[ax1, ax2], location='right', label='Magnitude')
    fig.colorbar(im3, ax=[ax3], location='right', label='Difference')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Open with PIL and save as compressed JPEG
    img = Image.open(buf)
    img = img.convert('RGB')
    img.save(cache_path, 'JPEG', quality=85, optimize=True)
    buf.close()



# tensor helpers

def lens_to_mask(
    t: Int[torch.Tensor, "b"],
    length: int | None = None
) -> Bool[torch.Tensor, "b n"]: 

    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device = t.device)
    return seq < t[..., None]


def l2_grad_norm(model: torch.nn.Module):
    return torch.cat([p.grad.data.flatten() for p in model.parameters() if p.grad is not None]).norm(2)


def nearest_interpolate_tensor(tensor, new_size):
    # Add two dummy dimensions to make it [1, 1, n, d]
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    
    # Interpolate
    interpolated = F.interpolate(tensor, size=(new_size, tensor.shape[-1]), mode='nearest')
    
    # Remove the dummy dimensions
    interpolated = interpolated.squeeze(0).squeeze(0)
    
    return interpolated


def linear_interpolate_tensor(tensor, new_size):
    # Assumes input tensor shape is [n, d]
    # Rearrange tensor to shape [1, d, n] to prepare for linear interpolation
    tensor = tensor.transpose(0, 1).unsqueeze(0)
    
    # Interpolate along the length dimension (last dimension) using linear interpolation.
    # align_corners=True preserves the boundary values; adjust this flag if needed.
    interpolated = F.interpolate(tensor, size=new_size, mode='linear', align_corners=True)
    
    # Restore the tensor to shape [new_size, d]
    return interpolated.squeeze(0).transpose(0, 1)


# f0 helpers

def post_process_f0(f0, sample_rate, hop_length, n_frames, silence_front=0.0, cut_last=True):
    """
    Post-process the extracted f0 to align with Mel spectrogram frames.

    Args:
        f0 (numpy.ndarray): Extracted f0 array.
        sample_rate (int): Sample rate of the audio.
        hop_length (int): Hop length used during processing.
        n_frames (int): Total number of frames (for alignment).
        silence_front (float): Seconds of silence to remove from the front.

    Returns:
        numpy.ndarray: Processed f0 array aligned with Mel spectrogram frames.
    """
    # Calculate number of frames to skip based on silence_front
    start_frame = int(silence_front * sample_rate / hop_length)
    real_silence_front = start_frame * hop_length / sample_rate
    # Assuming silence_front has been handled during RMVPE inference if needed

    # Handle unvoiced frames by interpolation
    uv = f0 == 0
    if np.any(~uv):
        f0_interp = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        f0[uv] = f0_interp
    else:
        # If no voiced frames, set all to zero
        f0 = np.zeros_like(f0)

    # Align with hop_length frames
    origin_time = 0.01 * np.arange(len(f0))  # Placeholder: Adjust based on RMVPE's timing
    target_time = hop_length / sample_rate * np.arange(n_frames - start_frame)
    f0 = np.interp(target_time, origin_time, f0)
    uv = np.interp(target_time, origin_time, uv.astype(float)) > 0.5
    f0[uv] = 0

    # Pad the silence_front if needed
    f0 = np.pad(f0, (start_frame, 0), mode='constant')

    if cut_last:
        return f0[:-1]
    else:
        return f0
    
# pyworld
def get_f0_pw(audio, sr, time_step, f0_min, f0_max):
    pw_pre_f0, times = pw.dio(
        audio.astype(np.double), sr,
        f0_floor=f0_min, f0_ceil=f0_max,
        frame_period=time_step*1000)    # raw pitch extractor
    pw_post_f0 = pw.stonemask(audio.astype(np.double), pw_pre_f0, times, sr)  # pitch refinement
    pw_post_f0[pw_post_f0==0] = np.nan
    pw_post_f0 = slide_nanmedian(pw_post_f0, 3)
    return pw_post_f0

# parselmouth
def get_f0_pm(audio, sr, time_step, f0_min, f0_max):
    pmac_pitch = pm.Sound(audio, sampling_frequency=sr).to_pitch_ac(
        time_step=time_step, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max,
        very_accurate=True, octave_jump_cost=0.5)
    pmac_f0 = pmac_pitch.selected_array['frequency']
    pmac_f0[pmac_f0==0] = np.nan
    pmac_f0 = slide_nanmedian(pmac_f0, 3)
    return pmac_f0

from numba import njit
@njit
def slide_nanmedian(signals=np.array([]), win_length=3):
    """Filters a sequence, ignoring nan values

    Arguments
        signals (numpy.ndarray (shape=(time)))
            The signals to filter
        win_length
            The size of the analysis window

    Returns
        filtered (numpy.ndarray (shape=(time)))
    """
    # Output buffer
    filtered = np.empty_like(signals)

    # Loop over frames
    for i in range(signals.shape[0]):

        # Get analysis window bounds
        start = max(0, i - win_length // 2)
        end = min(signals.shape[0], i + win_length // 2 + 1)

        # Apply filter to window
        filtered[i] = np.nanmedian(signals[start:end])

    return filtered


def f0_ensemble(rmvpe_f0, pw_f0, pmac_f0):
    trunc_len = len(rmvpe_f0)
    pw_f0 = pw_f0[:trunc_len]
    # pad pmac_f0
    pmac_f0 = np.concatenate(
        [pmac_f0, np.full(len(pw_f0)-len(pmac_f0), np.nan, dtype=pmac_f0.dtype)])

    stack_f0 = np.stack([pw_f0, pmac_f0, rmvpe_f0], axis=0)

    meadian_f0 = np.nanmedian(stack_f0, axis=0)
    nan_nums = np.sum(np.isnan(stack_f0), axis=0)
    meadian_f0[nan_nums>=2] = np.nan

    slide_meadian_f0 = slide_nanmedian(meadian_f0, 41)

    f0_dev = np.abs(meadian_f0-slide_meadian_f0)
    meadian_f0[f0_dev>96] = slide_meadian_f0[f0_dev>96]

    nan1_f0_min = np.nanmin(stack_f0[:, nan_nums==1], axis=0)
    nan1_f0_max = np.nanmax(stack_f0[:, nan_nums==1], axis=0)

    nan1_f0 = np.where(
        np.abs(nan1_f0_min-slide_meadian_f0[nan_nums==1])<np.abs(nan1_f0_max-slide_meadian_f0[nan_nums==1]),
        nan1_f0_min, nan1_f0_max)
    meadian_f0[nan_nums==1] = nan1_f0

    meadian_f0 = slide_nanmedian(meadian_f0, 3)
    meadian_f0[nan_nums>=2] = np.nan
    meadian_f0[np.isnan(meadian_f0)] = 0

    return meadian_f0


def f0_ensemble_light(rmvpe_f0, pw_f0, pmac_f0, rms=None, rms_threshold=0.05):
    """
    A lighter version of f0 ensemble that preserves RMVPE's expressiveness.
    Only applies corrections when RMVPE shows abnormalities.
    
    Args:
        rmvpe_f0 (numpy.ndarray): F0 from RMVPE
        pw_f0 (numpy.ndarray): F0 from WORLD
        pmac_f0 (numpy.ndarray): F0 from Parselmouth
        rms (numpy.ndarray, optional): RMS energy values, used to detect voiced segments
        rms_threshold (float, optional): Threshold for RMS to consider a segment as voiced
        
    Returns:
        numpy.ndarray: Corrected F0 values
    """
    trunc_len = len(rmvpe_f0)
    pw_f0 = pw_f0[:trunc_len]
    
    # Pad pmac_f0 if needed
    pmac_f0 = np.concatenate(
        [pmac_f0, np.full(max(0, len(pw_f0)-len(pmac_f0)), np.nan, dtype=pmac_f0.dtype)])
    
    # Create a copy of rmvpe_f0 to preserve most of its values
    corrected_f0 = rmvpe_f0.copy()
    
    # Stack all F0 values
    stack_f0 = np.stack([pw_f0, pmac_f0, rmvpe_f0], axis=0)
    
    # Count non-NaN values for each frame
    valid_count = np.sum(~np.isnan(stack_f0), axis=0)
    
    # Identify frames where RMVPE shows zero but other methods detect pitch
    zero_rmvpe_mask = (rmvpe_f0 == 0)
     
    # For frames where RMVPE is zero but at least one other method has a valid F0
    # and there's voice activity (if RMS is provided)
    other_methods_valid = ((~np.isnan(pw_f0) & (pw_f0 > 0)) | 
                           (~np.isnan(pmac_f0) & (pmac_f0 > 0)))
    
    correction_mask = zero_rmvpe_mask & other_methods_valid
    
    # If RMS is provided, only correct frames with voice activity
    if rms is not None:
        voice_activity = rms > rms_threshold
        correction_mask = correction_mask & voice_activity
    
    # For frames needing correction, use median of available values
    if np.any(correction_mask):
        # For each frame needing correction, calculate median of non-NaN values
        for i in np.where(correction_mask)[0]:
            valid_values = stack_f0[:, i][~np.isnan(stack_f0[:, i]) & (stack_f0[:, i] > 0)]
            if len(valid_values) > 0:
                corrected_f0[i] = np.median(valid_values)
    
    # Handle any remaining NaN values
    corrected_f0[np.isnan(corrected_f0)] = 0
    
    return corrected_f0


# progress bar helper

class CustomProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.step_start_time = None
        self.total_steps = None

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.start_time = time.time()
        self.total_steps = trainer.max_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        
        current_step = trainer.global_step
        total_steps = self.total_steps

        # Calculate elapsed time since training started
        elapsed_time = time.time() - self.start_time
        
        # Estimate average step time and remaining time
        average_step_time = elapsed_time / current_step if current_step > 0 else 0
        remaining_steps = total_steps - current_step
        remaining_time = average_step_time * remaining_steps if total_steps > 0 else 0

        # Format times with no leading zeros for hours
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours}:{minutes:02d}:{seconds:02d}"

        elapsed_time_str = format_time(elapsed_time)
        remaining_time_str = format_time(remaining_time)

        # Update the progress bar with loss, elapsed time, remaining time, and remaining steps
        self.train_progress_bar.set_postfix({
            "loss": f"{outputs['loss'].item():.4f}",
            "elapsed_time": elapsed_time_str + "/" + remaining_time_str,
            "remaining_steps": str(remaining_steps) + "/" + str(total_steps)
        })


# state dict helpers

def load_state_dict(model, state_dict, strict=False):
    """Load state dict while handling 'model.' prefix"""
    if any(k.startswith('model.') for k in state_dict.keys()):
        # Remove 'model.' prefix
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    return model.load_state_dict(state_dict, strict=strict)