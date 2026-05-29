# Bielik Supervised Fine-Tuning — HPC Eagle/Proxima

Runs Supervised Fine-Tuning (SFT) on the **Bielik-11B-v3.0-Instruct** model using QLoRA and the HuggingFace `trl` library on a single H100 GPU (or similar).

---

## Setup (one-time per user)

SSH into the cluster and add your project path to `~/.bashrc`:

```bash
echo 'export PROJECT_PATH="/mnt/storage_6/project_data/YOUR_GRANT"' >> ~/.bashrc
source ~/.bashrc

```

Replace `YOUR_GRANT` with your actual grant ID (e.g., `pl0966-02`).

## API Tokens & Configuration (one-time per user)

The Bielik model is gated, and training requires logging to Weights & Biases (W&B). You need tokens for both.

1. **HuggingFace:** Request access at [speakleash/Bielik-11B-v3.0-Instruct](https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct) and generate a write-access token at your [settings page](https://huggingface.co/settings/tokens).
2. **Weights & Biases:** Create an account at [wandb.ai](https://www.google.com/search?q=https://wandb.ai/) and copy your API key from your [authorizations page](https://wandb.ai/authorize).
3. Create a `.env` file in the `supervised-fine-tuning` directory and populate it:

```bash
cat <<EOF > $PROJECT_PATH/supervised-fine-tuning/.env
HF_TOKEN="your_hf_token_here"
WANDB_API_KEY="your_wandb_api_key_here"
PUSH_TO_HUB="true"
LOG_TO_WANDB="true"
QUANTIZATION="none" # Use 'none' for H100, or '4b'/'8b' for smaller GPUs
MAX_TRAIN_SAMPLES="0" # 0 means use the full dataset
EOF

```

---

## Copying Files to the Cluster

From your local machine, transfer the project files to the cluster:

```bash
cd /mnt/e/Projects/Pycharm/llm-fine-tuning/supervised-fine-tuning # (Your local project directory)
scp -i ~/.ssh/id_YOUR_ID -r ./* username@eagle.man.poznan.pl:/mnt/storage_6/project_data/YOUR_GRANT/supervised-fine-tuning

```

---

## Running the Setup

If you are running the fine-tuning for the first time, you must install the dependencies and create a virtual environment.

```bash
cd $PROJECT_PATH/supervised-fine-tuning
bash setup.sh

```

---

## Running the Training

Navigate to the fine-tuning directory and submit the SLURM job:

```bash
cd $PROJECT_PATH/supervised-fine-tuning
sbatch run_training.sh

```

---

## Monitoring

**Job Status & Logs:**

```bash
# Check job status in the queue
squeue -u $USER

# Follow standard output in real-time (replace JOBID with your job number)
tail -f logs/JOBID.out

# Check for critical errors
cat logs/JOBID.err

```

**Training Metrics:**
Once the job starts, open your [Weights & Biases Dashboard](https://www.google.com/search?q=https://wandb.ai/) to track the training loss, learning rate, and evaluation metrics in real-time.

---

## Project Structure

```text
$PROJECT_PATH/
├── supervised-fine-tuning/
│       ├── run_training.sh        # SLURM job submission script
│       ├── setup.sh               # Script for venv creation and dependency installation
│       ├── train.py               # Main SFT training script (TRL, PEFT, Datasets)
│       ├── config.py              # Centralized configuration class (hyperparameters)
│       ├── .env                   # Environment variables (Tokens, Flags)
│       └── requirements.txt       # Dependencies (torch, transformers, trl, peft, etc.)
├── logs/                          # Directory for SLURM output (.out) and error (.err) files
├── hf_cache/                      # Local HuggingFace model cache (prevents redownloads)
└── venv/                          # Shared Python virtual environment

```

---

## Troubleshooting

* **Job pending (`PD`):** Normal behavior, waiting for a free GPU partition. Check status with `squeue -u $USER`.
* **`OutOfMemoryError (OOM)`:** The model + dataset exceeded VRAM or System RAM.
* *Fix:* If it crashes during loading (System RAM), increase `#SBATCH --mem` in `run_training.sh`. If it crashes during training (GPU VRAM), reduce `BATCH_SIZE` in `config.py` or enable 4-bit quantization in `.env`.


* **`KeyError` during dataset loading:** Make sure you are using the correct `split` name ('train', 'test', or 'validation') matching the specific dataset in `train.py`.
* **W&B Sync Errors:** Ensure compute nodes have outbound internet access or consider running `wandb offline` and syncing manually later.