#!/bin/bash

# Default values
DEFAULT_DATA_DIR="data/finetune"
DEFAULT_NUM_WORKERS="1"

# Parse command line arguments
DATA_DIR="${1:-$DEFAULT_DATA_DIR}"
NUM_WORKERS_PER_DEVICE="${2:-$DEFAULT_NUM_WORKERS}"

# Execute the script with the provided or default values
echo "Using DATA_DIR: $DATA_DIR"
echo "Using NUM_WORKERS_PER_DEVICE: $NUM_WORKERS_PER_DEVICE"

python scripts/prepare_data_meta.py --data-dir $DATA_DIR
python scripts/prepare_mel.py --data-dir $DATA_DIR
python scripts/prepare_rms.py --data-dir $DATA_DIR
python scripts/prepare_f0.py --data-dir $DATA_DIR --num-workers-per-device $NUM_WORKERS_PER_DEVICE
python scripts/prepare_cvec.py --data-dir $DATA_DIR --num-workers-per-device $NUM_WORKERS_PER_DEVICE
python scripts/prepare_whisper.py --data-dir $DATA_DIR --num-workers-per-device $NUM_WORKERS_PER_DEVICE

python scripts/combine_features.py --data-dir $DATA_DIR