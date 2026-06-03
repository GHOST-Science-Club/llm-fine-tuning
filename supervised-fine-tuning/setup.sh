#!/bin/bash

module load python/3.11.9-gcc-11.5.0-5l7rvgy

# Load .env from current directory to pick up GRANT (and other vars)
if [ -f ".env" ]; then
    set -a; source ".env"; set +a
fi
if [ -n "$PROJECT_PATH" ]; then
    PROJECT="$PROJECT_PATH"
elif [ -n "$GRANT" ]; then
    PROJECT="/mnt/storage_6/project_data/$GRANT"
else
    echo "ERROR: Set GRANT in .env or PROJECT_PATH in ~/.bashrc"
    exit 1
fi

SFT_DIR="$PROJECT/supervised-fine-tuning"

export TMPDIR="$PROJECT/pip_tmp"
export PIP_CACHE_DIR="$PROJECT/pip_cache"

# Set up essential directories
mkdir -p "$TMPDIR"
mkdir -p "$PIP_CACHE_DIR"
mkdir -p "$SFT_DIR/logs"
mkdir -p "$SFT_DIR/models"

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
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r "$SFT_DIR/requirements.txt"

echo "Setup complete."