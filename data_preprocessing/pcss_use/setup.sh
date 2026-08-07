#!/bin/bash

module load python/3.11.9-gcc-11.5.0-5l7rvgy


if [ -z "$GRANT" ]; then
    echo "ERROR: Set GRANT in your environment before submitting (see comment above)."
    exit 1
fi

PROJECT="/mnt/storage_3/home/$USER/$GRANT/project_data/$USER/llm-fine-tuning"

PREPROCESSING="$PROJECT/data_preprocessing"

export TMPDIR="$PROJECT/pip_tmp"

mkdir -p "$TMPDIR"
mkdir -p "$PREPROCESSING/logs"


echo "Starting setup in directory: $PWD"
echo "Using project directory: $PROJECT"

if [ ! -d "$PROJECT/../../venv" ]; then
    echo "No venv environment found. Creating a new one..."
    python3 -m venv "$PROJECT/../../venv"
    source "$PROJECT/../../venv/bin/activate"

    echo "Installing libraries (this may take a few minutes)..."
    pip install --upgrade pip
    pip install -r "$PREPROCESSING/requirements.txt"
    pip install vllm

    echo "Installation complete."
else
    echo "venv environment already exists."
fi