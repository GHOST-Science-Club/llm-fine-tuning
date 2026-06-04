#!/bin/bash
#SBATCH --job-name=bielik_sft
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=proxima
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160GB
#SBATCH --time=24:00:00

module load python/3.11.9-gcc-11.5.0-5l7rvgy

# Load .env from submission directory (sbatch must be run from SFT_DIR)
if [ -f ".env" ]; then
    set -a; source ".env"; set +a
else
    echo "WARNING: .env file not found. If the model is gated, it will fail."
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

export HF_HOME="$PROJECT/hf_cache"
mkdir -p "$HF_HOME"

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