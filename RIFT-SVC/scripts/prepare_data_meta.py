"""
prepare_data_meta.py

This script scans a preprocessed audio dataset, gathers meta-information about speakers and their audio files,
and splits the data into training and testing sets based on specified options. The resulting meta-information
is saved in a JSON file.

Usage:
    python prepare_data_meta.py --data-dir PATH_TO_DATASET [OPTIONS]

Options:
    --data-dir                 Path to the preprocessed dataset directory. (Required)
    --split-type               Type of data split: 'random' or 'stratified'. (Default: 'random')
    --num-test                 Number of testing samples. (Default: 20)
    --num-test-per-speaker     Number of testing samples per speaker. (Default: 1)
    --seed                     Random seed for reproducibility. (Default: 42)
    --only-include-speakers     Only include these speakers in the meta-information. (Default: None)
"""
import json
import random
import sys
from pathlib import Path

import click


def gather_audio_files(data_dir):
    """
    Traverse the data directory and map speakers to their corresponding audio files.

    Args:
        data_dir (Path): Path to the dataset directory.

    Returns:
        dict: Mapping from speaker IDs to a list of their audio file stems.
    """
    speaker_to_files = {}
    for speaker_dir in data_dir.iterdir():
        if speaker_dir.is_dir():
            speaker_id = speaker_dir.name
            audio_files = sorted([file.stem for file in speaker_dir.glob('*.wav')])
            if audio_files:
                speaker_to_files[speaker_id] = audio_files
    return speaker_to_files


def perform_random_split(speaker_to_files, num_test, seed):
    """
    Perform a random split of the dataset into training and testing sets.

    Args:
        speaker_to_files (dict): Mapping of speakers to their audio files.
        num_test (int): Total number of testing samples.
        seed (int): Random seed.

    Returns:
        tuple: (train_audios, test_audios)
    """
    all_files = []
    for speaker, files in speaker_to_files.items():
        for file in files:
            all_files.append({"speaker": speaker, "file_name": file})

    if num_test > len(all_files):
        click.echo(
            f"Error: Requested number of testing samples ({num_test}) exceeds total available files ({len(all_files)}).",
            err=True
        )
        sys.exit(1)

    random.seed(seed)
    test_audios = random.sample(all_files, num_test)

    test_set = set((item['speaker'], item['file_name']) for item in test_audios)
    train_audios = [item for item in all_files if (item['speaker'], item['file_name']) not in test_set]

    return train_audios, test_audios


def perform_stratified_split(speaker_to_files, num_test_per_speaker, seed, only_include_speakers=None):
    """
    Perform a stratified split of the dataset into training and testing sets, ensuring each speaker has
    a specified number of testing samples.

    Args:
        speaker_to_files (dict): Mapping of speakers to their audio files.
        num_test_per_speaker (int): Number of testing samples per speaker.
        seed (int): Random seed.
        only_include_speakers (list): Only include these speakers in the meta-information.
    Returns:
        tuple: (train_audios, test_audios)
    """
    excluded_speakers = [
        "gtsinger-DE",
        "gtsinger-ES",
        "gtsinger-FR",
        "gtsinger-IT",
        "gtsinger-KO",
        "gtsinger-RU"
    ] + [
        "Tenor",
        "Bass",
        "ManRaw"
    ]

    if only_include_speakers:
        only_include_speakers = only_include_speakers.split(',')
        only_include_speakers = [speaker.strip() for speaker in only_include_speakers]
        # exclude speakers not in only_include_speakers
        excluded_speakers += [speaker for speaker in speaker_to_files.keys() if speaker not in only_include_speakers]

    random.seed(seed)
    train_audios = []
    test_audios = []

    for speaker, files in speaker_to_files.items():
        # Check if any excluded speaker substring is in the current speaker ID
        if any(excluded in speaker for excluded in excluded_speakers):
            # Assign all files to training set
            for file in files:
                train_audios.append({"speaker": speaker, "file_name": file})
            continue  # Skip to the next speaker

        if len(files) < num_test_per_speaker:
            click.echo(
                f"Warning: Speaker '{speaker}' has only {len(files)} files, less than the requested {num_test_per_speaker} testing samples.",
                err=True
            )
            test_files = files.copy()  # Take all available files
        else:
            test_files = random.sample(files, num_test_per_speaker)

        # Assign to testing set
        for file in test_files:
            test_audios.append({"speaker": speaker, "file_name": file})

        # Assign remaining to train set
        for file in files:
            if file not in test_files:
                train_audios.append({"speaker": speaker, "file_name": file})

    return train_audios, test_audios


def generate_meta_info(speaker_to_files, split_type, num_test, num_test_per_speaker, seed, only_include_speakers=None):
    """
    Generate the meta-information dictionary based on the split type.

    Args:
        speaker_to_files (dict): Mapping of speakers to their audio files.
        split_type (str): 'random' or 'stratified'.
        num_test (int): Number of testing samples for 'random' split.
        num_test_per_speaker (int): Number of testing samples per speaker for 'stratified' split.
        seed (int): Random seed.

    Returns:
        dict: Meta-information containing speakers, train_audios, and test_audios.
    """
    speakers = sorted(speaker_to_files.keys())

    if split_type == 'random':
        train_audios, test_audios = perform_random_split(speaker_to_files, num_test, seed)
    elif split_type == 'stratified':
        train_audios, test_audios = perform_stratified_split(speaker_to_files, num_test_per_speaker, seed, only_include_speakers)
    else:
        raise ValueError("Invalid split_type. Choose 'random' or 'stratified'.")

    meta_info = {
        "speakers": speakers,
        "train_audios": train_audios,
        "test_audios": test_audios
    }

    return meta_info


@click.command()
@click.option(
    '--data-dir',
    type=click.Path(exists=True, file_okay=False, readable=True),
    required=True,
    help='Path to the preprocessed dataset directory.'
)
@click.option(
    '--split-type',
    type=click.Choice(['random', 'stratified'], case_sensitive=False),
    default='random',
    show_default=True,
    help="Type of data split: 'random' or 'stratified'."
)
@click.option(
    '--num-test',
    type=int,
    default=20,
    show_default=True,
    help="Number of testing samples for 'random' split."
)
@click.option(
    '--num-test-per-speaker',
    type=int,
    default=1,
    show_default=True,
    help="Number of testing samples per speaker for 'stratified' split."
)
@click.option(
    '--only-include-speakers',
    type=str,
    default=None,
    show_default=True,
    help="Only include these speakers in the meta-information."
)
@click.option(
    '--seed',
    type=int,
    default=42,
    show_default=True,
    help="Random seed for reproducibility."
)
def main(data_dir, split_type, num_test, num_test_per_speaker, only_include_speakers, seed):
    """
    Generate meta-information for the preprocessed audio dataset.

    The meta-information includes a list of speakers, training audio files, and testing audio files.
    The testing set can be generated either by randomly selecting a specified number of files or by
    selecting a specified number of files from each speaker (stratified).
    """
    data_dir = Path(data_dir)
    speaker_to_files = gather_audio_files(data_dir)

    if not speaker_to_files:
        click.echo(f"No audio files found in the data directory '{data_dir}'.", err=True)
        sys.exit(1)

    meta_info = generate_meta_info(
        speaker_to_files,
        split_type=split_type.lower(),
        num_test=num_test,
        num_test_per_speaker=num_test_per_speaker,
        seed=seed,
        only_include_speakers=only_include_speakers
    )

    output_file = Path(data_dir) / "meta_info.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(meta_info, f, ensure_ascii=False, indent=4)
        click.echo(f"Meta-information successfully saved to '{output_file}'.")
    except Exception as e:
        click.echo(f"Error writing to output file '{output_file}': {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()