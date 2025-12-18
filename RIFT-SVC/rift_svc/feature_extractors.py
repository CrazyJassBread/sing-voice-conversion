import torch
from torch import nn
from jaxtyping import Float
from librosa.filters import mel as librosa_mel_fn
from transformers import HubertModel


def dynamic_range_compression_torch(
        x: Float[torch.Tensor, "n_mels mel_len"],
        C: float = 1,
        clip_val: float = 1e-5
) -> Float[torch.Tensor, "n_mels mel_len"]:
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(
        magnitudes: Float[torch.Tensor, "n_mels mel_len"]
) -> Float[torch.Tensor, "n_mels mel_len"]:
    return dynamic_range_compression_torch(magnitudes)


mel_basis_cache = {}
hann_window_cache = {}


def get_mel_spectrogram(
    y: Float[torch.Tensor, "n"],
    n_fft: int = 2048,
    num_mels: int = 128,
    sampling_rate: int = 44100,
    hop_size: int = 512,
    win_size: int = 2048,
    fmin: int = 40,
    fmax: int | None = 16000,
    center: bool = False,
) -> Float[torch.Tensor, "n_mels mel_len"]:
    """
    Calculate the mel spectrogram of an input signal.
    This function uses slaney norm for the librosa mel filterbank (using librosa.filters.mel) and uses Hann window for STFT (using torch.stft).

    Args:
        y (torch.Tensor): Input signal with shape (n,).
        n_fft (int, optional): FFT size. Defaults to 1024.
        num_mels (int, optional): Number of mel bins. Defaults to 128.
        sampling_rate (int, optional): Sampling rate of the input signal. Defaults to 44100.
        hop_size (int, optional): Hop size for STFT. Defaults to 256.
        win_size (int, optional): Window size for STFT. Defaults to 1024.
        fmin (int, optional): Minimum frequency for mel filterbank. Defaults to 0.
        fmax (int | None, optional): Maximum frequency for mel filterbank. If None, defaults to sr/2.0. Defaults to None.
        center (bool, optional): Whether to pad the input to center the frames. Defaults to False.

    Returns:
        torch.Tensor: Mel spectrogram with shape (n_mels, mel_len).
    """
    if torch.min(y) < -1.0:
        print(f"[WARNING] Min value of input waveform signal is {torch.min(y)}")
    if torch.max(y) > 1.0:
        print(f"[WARNING] Max value of input waveform signal is {torch.max(y)}")

    device = y.device
    key = f"{n_fft}_{num_mels}_{sampling_rate}_{hop_size}_{win_size}_{fmin}_{fmax}_{device}"

    if key not in mel_basis_cache:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)
        hann_window_cache[key] = torch.hann_window(win_size).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(
        y.unsqueeze(1), (padding, padding), mode="reflect"
    ).squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = spectral_normalize_torch(mel_spec)

    return mel_spec


class RMSExtractor(nn.Module):
    def __init__(self, hop_length=512, window_length=2048):
        """
        Initializes the RMSExtractor with the specified hop_length.

        Args:
            hop_length (int): Number of samples between successive frames.
        """
        super(RMSExtractor, self).__init__()
        self.hop_length = hop_length
        self.window_length = window_length

    def forward(self, inp):
        """
        Extracts RMS energy from the input audio tensor.

        Args:
            inp (Tensor): Audio tensor of shape (batch, samples).

        Returns:
            Tensor: RMS energy tensor of shape (batch, frames).
        """
        # Square the audio signal
        audio_squared = inp ** 2

        # Use the same padding as mel spectrogram
        padding = (self.window_length - self.hop_length) // 2
        audio_padded = torch.nn.functional.pad(
            audio_squared, (padding, padding), mode='reflect'
        )

        # Unfold to create frames with window_length instead of hop_length
        frames = audio_padded.unfold(1, self.window_length, self.hop_length)  # Shape: (batch, frames, window_length)

        # Compute mean energy per frame
        mean_energy = frames.mean(dim=-1)  # Shape: (batch, frames)

        # Compute RMS by taking square root
        rms = torch.sqrt(mean_energy)  # Shape: (batch, frames)

        return rms


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)