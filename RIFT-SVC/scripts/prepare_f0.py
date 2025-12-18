import json
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import click
import torch
import torchaudio
from functools import partial
import multiprocessing

from multiprocessing_utils import run_parallel, get_device
from rift_svc.rmvpe.inference import RMVPE

RMVPE_HOP_LENGTH = 160

def get_f0_model(model_path):
    """
    Lazy-load the RMVPE model, loading and caching it on its first invocation within each process.
    """
    if not hasattr(get_f0_model, "model"):
        device = get_device()
        model = RMVPE(model_path=model_path, hop_length=RMVPE_HOP_LENGTH, device=device)
        get_f0_model.model = model
        get_f0_model.device = device
    return get_f0_model.model, get_f0_model.device

def process_f0(audio, data_dir, model_path, hop_length, sample_rate, overwrite, verbose):
    """
    Extract the f0 for a single audio file and save it as a .f0.pt file.
    """
    speaker = audio.get('speaker')
    file_name = audio.get('file_name')
    if not speaker or not file_name:
        if verbose:
            click.echo(f"Skipping invalid entry: {audio}")
        return
    wav_path = Path(data_dir) / speaker / f"{file_name}.wav"
    f0_path = Path(data_dir) / speaker / f"{file_name}.f0.pt"
    
    if f0_path.is_file() and not overwrite:
        if verbose:
            click.echo(f"Skipping existing f0 file: {f0_path}")
        return
    if not wav_path.is_file():
        if verbose:
            click.echo(f"Warning: WAV file not found: {wav_path}")
        return
    try:
        waveform, sr = torchaudio.load(str(wav_path))
        model, device = get_f0_model(model_path)
        waveform = waveform.to(device)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2 and waveform.shape[0] != 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        
        with torch.no_grad():
            f0 = model.infer_from_audio(
                waveform,
                sample_rate=sample_rate,
                device=device,
                thred=0.03,
                use_viterbi=False
            )
        n_frames = int(waveform.shape[-1] // hop_length) + 1

        from rift_svc.utils import post_process_f0
        f0_processed = post_process_f0(
            f0=f0,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_frames=n_frames,
            silence_front=0.0
        )
        f0_tensor = torch.from_numpy(f0_processed).float().cpu()
        torch.save(f0_tensor, f0_path)
        if verbose:
            click.echo(f"Saved f0: {f0_path}")
    except Exception as e:
        click.echo(f"Error processing {wav_path}: {e}")

@click.command()
@click.option(
    '--data-dir',
    type=click.Path(exists=True, file_okay=False, readable=True),
    required=True,
    help='Preprocessed dataset root directory.'
)
@click.option(
    '--model-path',
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=False,
    default="pretrained/rmvpe/model.pt",
    help='Pre-trained RMVPE model path.'
)
@click.option(
    '--hop-length',
    type=int,
    default=512,
    show_default=True,
    help='Hop length for f0 extraction.'
)
@click.option(
    '--sample-rate',
    type=int,
    default=44100,
    show_default=True,
    help='Audio sample rate (Hz).'
)
@click.option(
    '--num-workers',
    type=int,
    default=2,
    show_default=True,
    help='Number of parallel processes.'
)
@click.option(
    '--overwrite',
    is_flag=True,
    default=False,
    help='Whether to overwrite existing f0 files.'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='Whether to display detailed logs.'
)
def prepare_f0(data_dir, model_path, hop_length, sample_rate, num_workers, overwrite, verbose):
    """
    Extract the f0 for all audio files listed in meta_info.json and save them as .f0.pt files.
    """
    meta_info_path = Path(data_dir) / "meta_info.json"
    try:
        with open(meta_info_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        click.echo(f"Error reading meta_info.json: {e}")
        sys.exit(1)

    train_audios = meta.get('train_audios', [])
    test_audios = meta.get('test_audios', [])
    all_audios = train_audios + test_audios

    if not all_audios:
        click.echo("No audio files found in meta_info.json.")
        sys.exit(1)

    process_func = partial(
        process_f0,
        data_dir=data_dir,
        model_path=model_path,
        hop_length=hop_length,
        sample_rate=sample_rate,
        overwrite=overwrite,
        verbose=verbose
    )

    run_parallel(
        all_audios,
        process_func,
        num_workers=num_workers,
        desc="Extracting f0"
    )

    click.echo("f0 extraction complete.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    prepare_f0()