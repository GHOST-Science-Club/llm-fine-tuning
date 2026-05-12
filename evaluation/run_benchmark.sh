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

export HF_HOME="$PROJECT/hf_cache"
export TMPDIR="$PROJECT/pip_tmp"
export PIP_CACHE_DIR="$PROJECT/pip_cache"

mkdir -p $HF_HOME
mkdir -p $TMPDIR
mkdir -p $PIP_CACHE_DIR

echo "Starting setup in directory: $PWD"
echo "Using project directory: $PROJECT"

if [ ! -d "$PROJECT/venv" ]; then
    echo "No venv environment found. Creating a new one..."
    python3 -m venv $PROJECT/venv
    source $PROJECT/venv/bin/activate

    echo "Installing libraries (this may take a few minutes)..."
    pip install --upgrade pip
    pip install -r requirements.txt

    echo "Installation complete."
else
    echo "venv environment already exists. Activating..."
    source $PROJECT/venv/bin/activate
fi

echo "Starting evaluation..."
python3 $PROJECT/llm-fine-tuning/evaluation/run_eval.py

echo "Job completed."