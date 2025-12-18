import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import torch
import torchaudio
import click
from functools import partial

from rift_svc.feature_extractors import get_mel_spectrogram
from multiprocessing_utils import run_parallel, get_device

def process_audio(audio, data_dir, hop_length, n_mel_channels, sample_rate, verbose, overwrite, device):
    """
    Process a single audio file: read the WAV file, generate its Mel spectrogram,
    and save it as a .mel.pt file.
    """
    speaker = audio.get('speaker')
    file_name = audio.get('file_name')
    # Skip if information is incomplete
    if not speaker or not file_name:
        if verbose:
            click.echo(f"Skipping invalid entry: {audio}", err=True)
        return

    wav_path = Path(data_dir) / speaker / f"{file_name}.wav"
    mel_path = Path(data_dir) / speaker / f"{file_name}.mel.pt"

    if mel_path.is_file() and not overwrite:
        if verbose:
            click.echo(f"Skipping existing Mel spectrogram: {mel_path}", err=True)
        return

    if not wav_path.is_file():
        if verbose:
            click.echo(f"Warning: WAV file not found: {wav_path}", err=True)
        return

    try:
        waveform, sr = torchaudio.load(str(wav_path))
        # Ensure the correct shape
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        elif len(waveform.shape) == 2 and waveform.shape[0] != 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.to(device)

        mel = get_mel_spectrogram(
            waveform,
            hop_size=hop_length,
            num_mels=n_mel_channels,
            sampling_rate=sample_rate,
            n_fft=2048,
            win_size=2048,
            fmin=40,
            fmax=16000,
        )
        mel = mel.cpu()  # Move back to CPU for saving

        torch.save(mel, mel_path)

        if verbose:
            click.echo(f"Saved Mel spectrogram: {mel_path}")

    except Exception as e:
        click.echo(f"Error processing {wav_path}: {e}", err=True)

@click.command()
@click.option(
    '--data-dir',
    type=click.Path(exists=True, file_okay=False, readable=True),
    required=True,
    help='Root directory of the preprocessed dataset.'
)
@click.option(
    '--hop-length',
    type=int,
    default=512,
    show_default=True,
    help='Hop length for the Mel spectrogram.'
)
@click.option(
    '--n-mel-channels',
    type=int,
    default=128,
    show_default=True,
    help='Number of Mel channels.'
)
@click.option(
    '--sample-rate',
    type=int,
    default=44100,
    show_default=True,
    help='Target sample rate (Hz).'
)
@click.option(
    '--num-workers',
    type=int,
    default=4,
    show_default=True,
    help='Number of parallel processes.'
)
@click.option(
    '--overwrite',
    is_flag=True,
    default=False,
    help='Whether to overwrite existing Mel spectrogram files.'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='Whether to print detailed logs.'
)
def generate_mel_specs(data_dir, hop_length, n_mel_channels, sample_rate, num_workers, verbose, overwrite):
    """
    Generate Mel spectrograms for the audio files listed in meta_info.json and
    save them as .mel.pt files.
    """
    meta_info = Path(data_dir) / "meta_info.json"
    try:
        with open(meta_info, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        click.echo(f"Error reading meta_info.json: {e}", err=True)
        sys.exit(1)

    train_audios = meta.get('train_audios', [])
    test_audios = meta.get('test_audios', [])
    all_audios = (
        [{'type': 'train', 'index': i, **audio} for i, audio in enumerate(train_audios)] +
        [{'type': 'test', 'index': i, **audio} for i, audio in enumerate(test_audios)]
    )

    if not all_audios:
        click.echo("No audio files found in meta_info.json.", err=True)
        sys.exit(1)

    device = get_device()
    if verbose:
        click.echo(f"Using device: {device}")

    torch.set_grad_enabled(False)

    process_func = partial(
        process_audio,
        data_dir=data_dir,
        hop_length=hop_length,
        n_mel_channels=n_mel_channels,
        sample_rate=sample_rate,
        verbose=verbose,
        overwrite=overwrite,
        device=device
    )

    run_parallel(all_audios, process_func, num_workers=num_workers, desc="Generating Mel Spectrograms")
    click.echo("Mel spectrogram generation complete.")

if __name__ == "__main__":
    generate_mel_specs()