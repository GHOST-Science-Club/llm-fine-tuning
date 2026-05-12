#!/bin/bash

module load python/3.11.9-gcc-11.5.0-5l7rvgy
PROJECT="${PROJECT_PATH:-/mnt/storage_6/project_data/pl0966-02}"
EVALUATION="$PROJECT/evaluation"

export TMPDIR="$PROJECT/pip_tmp"
export PIP_CACHE_DIR="$PROJECT/pip_cache"

mkdir -p "$TMPDIR"
mkdir -p "$PIP_CACHE_DIR"


echo "Starting setup in directory: $PWD"
echo "Using project directory: $PROJECT"

if [ ! -d "$PROJECT/venv" ]; then
    echo "No venv environment found. Creating a new one..."
    python3 -m venv "$PROJECT/venv"
    source "$PROJECT/venv/bin/activate"

    echo "Installing libraries (this may take a few minutes)..."
    pip install --upgrade pip
    pip install -r "$EVALUATION/requirements.txt"

    echo "Installation complete."
else
    echo "venv environment already exists."
fi