#!/bin/bash
#SBATCH --job-name=bielik_eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=proxima
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --time=04:00:00

module load python/3.11.9-gcc-11.5.0-5l7rvgy

# Set PROJECT_PATH in your ~/.bashrc: export PROJECT_PATH="/mnt/storage_6/project_data/YOUR_GRANT"
PROJECT="${PROJECT_PATH:-/mnt/storage_6/project_data/pl0966-02}"
EVALUATION="$PROJECT/evaluation"

export HF_HOME="$PROJECT/hf_cache"
mkdir -p "$HF_HOME"

if [ ! -d "$PROJECT/venv" ]; then
    echo "ERROR: venv environment does not exist."
    echo "Please run setup.sh manually on the login node before submitting this job."
    exit 1
fi

echo "Activating venv..."
source "$PROJECT/venv/bin/activate"

echo "Starting evaluation..."
python3 "$EVALUATION/run_eval.py"

echo "Job completed."