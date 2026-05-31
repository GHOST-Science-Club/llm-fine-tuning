#!/bin/bash
#SBATCH --job-name=bielik_sft
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=proxima
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=24:00:00

module load python/3.11.9-gcc-11.5.0-5l7rvgy

# Set PROJECT_PATH in your ~/.bashrc: export PROJECT_PATH="/mnt/storage_6/project_data/YOUR_GRANT"
PROJECT="${PROJECT_PATH:-/mnt/storage_6/project_data/pl0966-02}"
SFT_DIR="$PROJECT/supervised-fine-tuning"

mkdir -p "$SFT_DIR/logs"

export HF_HOME="$PROJECT/hf_cache"
mkdir -p "$HF_HOME"

if [ -f "$SFT_DIR/.env" ]; then
    echo "Loading environment variables from .env file..."
    set -a
    source "$SFT_DIR/.env"
    set +a
else
    echo "WARNING: .env file not found. If the model is gated, it will fail."
fi

if [ ! -d "$PROJECT/venv" ]; then
    echo "ERROR: venv environment does not exist."
    echo "Please run setup.sh manually on the login node before submitting this job."
    exit 1
fi

echo "Activating venv..."
source "$PROJECT/venv/bin/activate"

echo "Starting supervised fine-tuning..."
python3 "$SFT_DIR/train.py"

echo "Job completed."