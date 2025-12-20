import os
os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"

import argparse
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml


def calculate_semitone_shift(source_f0, reference_f0):
    """Calculate the semitone shift between source and reference F0."""
    if isinstance(source_f0, torch.Tensor):
        source_f0 = source_f0.cpu().numpy()
    if isinstance(reference_f0, torch.Tensor):
        reference_f0 = reference_f0.cpu().numpy()

    src_valid = source_f0[source_f0 > 1]
    ref_valid = reference_f0[reference_f0 > 1]

    if len(src_valid) == 0 or len(ref_valid) == 0:
        return 0.0

    log_src = np.log(src_valid)
    log_ref = np.log(ref_valid)

    mean_log_diff = np.mean(log_ref) - np.mean(log_src)

    return 12 * mean_log_diff / np.log(2)  # Removed the +4.5 offset


def get_f0_statistics(f0_sequence):
    """Get statistics of F0 sequence for adaptive processing."""
    if isinstance(f0_sequence, torch.Tensor):
        f0_sequence = f0_sequence.cpu().numpy()

    valid_f0 = f0_sequence[f0_sequence > 1]
    if len(valid_f0) == 0:
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}

    return {
        "mean": np.mean(valid_f0),
        "median": np.median(valid_f0),
        "std": np.std(valid_f0),
        "min": np.min(valid_f0),
        "max": np.max(valid_f0),
    }


def adaptive_pitch_shift_factor(voiced_f0_ori, low_threshold=120, high_threshold=205, min_factor=0.3, max_factor=1.0):
    """
    Calculate adaptive pitch shift factor based on original F0 characteristics.

    Args:
        voiced_f0_ori: Original F0 sequence
        low_threshold: F0 threshold below which to increase shift (Hz)
        high_threshold: F0 threshold above which to reduce shift (Hz)
        min_factor: Minimum reduction factor for high F0
        max_factor: Maximum factor (no reduction)

    Returns:
        Adaptive factor
    """
    f0_stats = get_f0_statistics(voiced_f0_ori)

    if f0_stats["mean"] == 0:
        return max_factor

    mean_f0 = f0_stats["mean"]

    # For high F0 (typically female voices), reduce pitch shift
    if mean_f0 > high_threshold:
        # Gradual reduction as F0 increases
        factor = max(min_factor, max_factor - (mean_f0 - high_threshold) / (560 - high_threshold))

    # For low F0 (typically male voices), allow more pitch shift
    elif mean_f0 < low_threshold:
        # Gradual increase as F0 decreases
        # factor = min(max_factor, max_factor + (low_threshold - mean_f0) / low_threshold * 0.8)
        factor = max_factor - (low_threshold - mean_f0) / low_threshold * 1.2  # * 0.8

    else:
        factor = max_factor

    return factor


def adjust_f0_semitones(f0_sequence, n_semitones):
    """Adjust F0 sequence by n semitones."""
    if abs(n_semitones) < 0.1:  # Skip very small adjustments
        return f0_sequence.clone()

    adjusted = f0_sequence.clone()
    factor = 2 ** (n_semitones / 12.0)
    valid_mask = adjusted > 1
    adjusted[valid_mask] = adjusted[valid_mask] * factor
    return adjusted

def semitone_map(x: float, threshold=7) -> int:
    if x >= threshold:
        return 12
    if x <= - threshold:
        return -12
    return 0


def preprocess_voice_conversion(
    voiced_f0_ori, voiced_f0_alt, shifted_f0_alt, enable_adaptive=True, max_shift_semitones=8, forch_pitch_shift=None
):
    """
    Main preprocessing function for voice conversion.

    Args:
        voiced_f0_ori: Original F0 sequence
        voiced_f0_alt: Alternative F0 sequence for reference
        shifted_f0_alt: Pre-shifted F0 sequence
        enable_adaptive: Whether to use adaptive pitch shifting
        max_shift_semitones: Maximum allowed pitch shift in semitones

    Returns:
        final_f0_alt: Processed F0 sequence
    """
    if forch_pitch_shift == None:
        # Calculate base pitch shift
        base_pitch_shift = calculate_semitone_shift(voiced_f0_alt, voiced_f0_ori)

        # Apply adaptive factor if enabled
        if enable_adaptive:
            adaptive_factor = adaptive_pitch_shift_factor(voiced_f0_ori)
            pitch_shift = base_pitch_shift * adaptive_factor + 3.5
        else:
            pitch_shift = base_pitch_shift

        # Clamp pitch shift to reasonable range
        pitch_shift = np.clip(pitch_shift, -max_shift_semitones, max_shift_semitones)

        print(f'auto predicted pitch shift: {pitch_shift}')
        # 离散化,以8semitone为界限做八度变化
        pitch_shift = semitone_map(pitch_shift)

    else:
        pitch_shift = forch_pitch_shift

    # Apply pitch adjustment
    final_f0_alt = adjust_f0_semitones(shifted_f0_alt, pitch_shift)


    return final_f0_alt, pitch_shift





