#!/bin/bash
# Launch training on the TPU VM.
# changed vs gpu/run_training.sh: no SLURM (#SBATCH) — on a TPU VM you just run
# the script directly; the TPU is already attached to the machine.
set -e
cd "$(dirname "$0")"

if [ -f ".env" ]; then
    set -a; source ".env"; set +a
else
    echo "WARNING: .env file not found. If the model is gated, it will fail."
fi

source venv/bin/activate

# PJRT is the runtime torch_xla uses to talk to the accelerator.
# Set PJRT_DEVICE=CPU to smoke-test the whole pipeline on any machine, without a TPU.
export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"

# Keep the HF model cache next to the code (mount a bigger data disk here if
# the model doesn't fit the TPU VM's boot disk).
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
mkdir -p "$HF_HOME" logs

echo "Starting supervised fine-tuning on device: $PJRT_DEVICE"
python3 train.py 2>&1 | tee "logs/train_$(date +%Y%m%d_%H%M%S).log"

echo "Job completed."
