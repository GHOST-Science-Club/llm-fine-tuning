#!/bin/bash

module load python/3.11.9-gcc-11.5.0-5l7rvgy
PROJECT="${PROJECT_PATH:-/mnt/storage_6/project_data/pl0966-02}"
SFT_DIR="$PROJECT/supervised-fine-tuning"

export TMPDIR="$PROJECT/pip_tmp"
export PIP_CACHE_DIR="$PROJECT/pip_cache"

mkdir -p "$TMPDIR"
mkdir -p "$PIP_CACHE_DIR"
mkdir -p "$SFT_DIR/logs"

echo "Starting setup in directory: $PWD"
echo "Using project directory: $PROJECT"

if [ ! -d "$PROJECT/venv" ]; then
    echo "No venv environment found. Creating a new one..."
    python3 -m venv "$PROJECT/venv"
else
    echo "venv environment already exists. Proceeding to update libraries..."
fi

source "$PROJECT/venv/bin/activate"

echo "Installing/Updating libraries (this may take a few minutes)..."
pip install --upgrade pip
pip install -r "$SFT_DIR/requirements.txt"

echo "Setup complete."