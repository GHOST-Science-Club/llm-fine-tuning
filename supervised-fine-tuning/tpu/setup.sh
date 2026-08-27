#!/bin/bash
# One-time setup on a Google Cloud TPU VM (or any machine for CPU smoke tests).
# changed vs gpu/setup.sh: no SLURM/module-load and no grant storage paths —
# a TPU VM is a plain machine you SSH into; everything lives next to the code.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d venv ]; then
    echo "Creating venv..."
    python3 -m venv venv
fi
source venv/bin/activate

echo "Installing/Updating libraries (this may take a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p logs models
echo "Setup complete."
