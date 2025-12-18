import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import multiprocessing
import click
import torch
import torchaudio
from functools import partial

from multiprocessing_utils import run_parallel, get_device
from rift_svc.feature_extractors import HubertModelWithFinalProj


def roll_pad(wav, shift):
    wav = torch.roll(wav, shift, dims=1)
    if shift > 0:
        wav[:, :shift] = 0
    else:
        wav[:, shift:] = 0
    return wav

CVEC_SAMPLE_RATE = 16000

def get_cvec_model(model_path):
    """
    Lazy-load the HuBERT model, loading and caching it on its first invocation in each process.
    """
    if not hasattr(get_cvec_model, "model"):
        device = get_device()
        model = HubertModelWithFinalProj.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        get_cvec_model.model = model
        get_cvec_model.device = device
    return get_cvec_model.model, get_cvec_model.device

def process_cvec(audio, data_dir, model_path, overwrite, verbose):
    """
    Extract the content vector for a single audio file and save it as a .cvec.pt file.
    """
    speaker = audio.get('speaker')
    file_name = audio.get('file_name')
    if not speaker or not file_name:
        if verbose:
            click.echo(f"Skipping invalid entry: {audio}")
        return
    
    wav_path = Path(data_dir) / speaker / f"{file_name}.wav"
    cvec_path = Path(data_dir) / speaker / f"{file_name}.cvec.pt"
    
    if cvec_path.is_file() and not overwrite:
        if verbose:
            click.echo(f"Skipping existing content vector: {cvec_path}")
        return
    
    if not wav_path.is_file():
        if verbose:
            click.echo(f"Warning: WAV file not found: {wav_path}")
        return
    
    try:
        waveform, sr = torchaudio.load(str(wav_path))
        model, device = get_cvec_model(model_path)
        waveform = waveform.to(device)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2 and waveform.shape[0] != 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != CVEC_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, CVEC_SAMPLE_RATE)

        with torch.no_grad():
            output = model(waveform)  # returns a dictionary containing "last_hidden_state"
            cvec = output["last_hidden_state"].squeeze(0).cpu()

            # Process the shifted waveform
            waveform_shifted = roll_pad(waveform, -160)
            output_shifted = model(waveform_shifted)
            cvec_shifted = output_shifted["last_hidden_state"].squeeze(0).cpu()

            n, d = cvec.shape
            cvec = torch.stack([cvec, cvec_shifted], dim=1).view(n * 2, d)
        
        torch.save(cvec, cvec_path)
        if verbose:
            click.echo(f"Saved content vector: {cvec_path}")
    except Exception as e:
        click.echo(f"Error processing {wav_path}: {e}")

@click.command()
@click.option(
    '--data-dir',
    type=click.Path(exists=True, file_okay=False, readable=True),
    required=True,
    help='Root directory of the preprocessed dataset.'
)
@click.option(
    '--model-path',
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=False,
    default="pretrained/content-vec-best",
    help='Path to the pre-trained HuBERT-like model.'
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
    help='Whether to overwrite existing content vectors.'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='Whether to display detailed logs.'
)
def prepare_contentvec(data_dir, model_path, num_workers, overwrite, verbose):
    """
    Extract content vectors from the audio files listed in meta_info.json and save them as .cvec.pt files.
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
    
    # Pass model_path as a parameter to the processing function
    process_func = partial(
        process_cvec,
        data_dir=data_dir,
        model_path=model_path,
        overwrite=overwrite,
        verbose=verbose
    )
    
    run_parallel(
        all_audios,
        process_func,
        num_workers=num_workers,
        desc="Extracting Content Vectors"
    )
    
    click.echo("Content vector extraction complete.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    prepare_contentvec()
