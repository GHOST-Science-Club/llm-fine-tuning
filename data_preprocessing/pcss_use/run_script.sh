#!/bin/bash
#SBATCH --job-name=vllm_server
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# #SBATCH --partition=proxima
# #SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1 --constraint=h100
#SBATCH --partition=tesla
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150GB
#SBATCH --time=04:00:00

module load python/3.11.9-gcc-11.5.0-5l7rvgy


if ! nvidia-smi -L | grep -q "GPU"; then
    echo "ERROR: No GPU available to this job. Exiting early."
    exit 1
fi

python3 -c "import torch; torch.cuda.init()"

# GRANT must be exported in your shell/SLURM environment before submitting
# (e.g. `export GRANT=pl0123-45`)
if [ -z "$GRANT" ]; then
    echo "ERROR: Set GRANT in your environment before submitting (see comment above)."
    exit 1
fi

PROJECT="/mnt/storage_3/home/$USER/$GRANT/project_data/$USER/llm-fine-tuning"
PREPROCESSING="$PROJECT/data_preprocessing"

set -a
source $PREPROCESSING/.env
set +a

export HF_HOME="$PROJECT/hf_cache"
mkdir -p "$HF_HOME"

if [ -f "$PREPROCESSING/.env" ]; then
    echo "Loading environment variables from .env file..."
    export $(grep '^HF_TOKEN=' "$PREPROCESSING/.env" | xargs)
else
    echo "WARNING: .env file not found. If the model is gated, it will fail."
fi

if [ ! -d "$PROJECT/../../venv" ]; then
    echo "ERROR: venv environment does not exist."
    echo "Please run setup.sh manually on the login node before submitting this job."
    exit 1
fi

echo "Activating venv..."
source "$PROJECT/../../venv/bin/activate"

# As Llama is gated, we need to authenticate with Hugging Face before starting the vLLM server.

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
if ! python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: PyTorch cannot see a CUDA GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES). Exiting early."
    exit 1
fi

hf auth login --token $HF_TOKEN

echo "Starting vLLM server..."
echo "Using model: $MODEL"
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --host 127.0.0.1 \
    --port 8080 \
    --gpu-memory-utilization=0.95 &
VLLM_PID=$!

echo "Waiting for vLLM server to become ready..."
until curl -s http://127.0.0.1:8080/v1/models > /dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server process died before becoming ready."
        exit 1
    fi
    sleep 1
done
echo "vLLM server is up."

echo "Starting pipeline..."
cd "$PROJECT" && python3 -m data_preprocessing.pipeline

echo "Stopping vLLM server..."
kill "$VLLM_PID"
wait "$VLLM_PID" 2>/dev/null

echo "Job completed."
exit 0