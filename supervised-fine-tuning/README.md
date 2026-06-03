# Bielik Supervised Fine-Tuning — HPC Eagle/Proxima

Runs Supervised Fine-Tuning (SFT) on the **Bielik-11B-v3.0-Instruct** model using QLoRA and the HuggingFace `trl` library on a single H100 GPU (or similar).

---

## Setup (one-time per user)

### 1. API Tokens & Configuration

The Bielik model is gated and training logs to Weights & Biases (W&B). You need tokens for both.

1. **HuggingFace:** Request access at [speakleash/Bielik-11B-v3.0-Instruct](https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct) and generate a token at your [settings page](https://huggingface.co/settings/tokens). The token needs **Write** access and **"Create and manage repos"** permission (required to push checkpoints during training).
2. **Weights & Biases:** Create an account at [wandb.ai](https://wandb.ai) and copy your API key from your [authorizations page](https://wandb.ai/authorize).

### 2. Create `.env`

Copy `.env.example` to `.env` in the `supervised-fine-tuning` directory and fill in all values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `GRANT` | **yes** | Your HPC grant ID (e.g. `pl0966-02`) — used to build the project path |
| `HF_TOKEN` | yes | HuggingFace token with Write + repo-create permissions |
| `WANDB_API_KEY` | yes (if logging) | W&B API key |
| `PUSH_TO_HUB` | no | `true` to push checkpoints to HF Hub during training |
| `LOG_TO_WANDB` | no | `true` to enable W&B logging |
| `QUANTIZATION` | no | `none` (default, for H100), `4b`, or `8b` |
| `MAX_TRAIN_SAMPLES` | no | Limit training examples; `0` = full dataset |


---

## Copying Files to the Cluster

From your local machine, transfer the project files to the cluster:

```bash
scp -i ~/.ssh/id_YOUR_ID -r ./* username@eagle.man.poznan.pl:/mnt/storage_6/project_data/YOUR_GRANT/supervised-fine-tuning
```

---

## Running the Setup

SSH into the cluster, navigate to the project directory, and run setup **once** from that directory (the scripts source `.env` from the current directory):

```bash
cd /mnt/storage_6/project_data/YOUR_GRANT/supervised-fine-tuning
bash setup.sh
```

This creates the Python venv and installs all dependencies.

---

## Running the Training

Submit the SLURM job **from the project directory** (the script sources `.env` from the current directory):

```bash
cd /mnt/storage_6/project_data/YOUR_GRANT/supervised-fine-tuning
sbatch run_training.sh
```

Checkpoints are saved to `models/bielik-tuning-<timestamp>/`.

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
Once the job starts, open your [Weights & Biases Dashboard](https://wandb.ai) to track the training loss, learning rate, and evaluation metrics in real-time.

---

## Project Structure

```text
supervised-fine-tuning/
├── train.py               # Main SFT training script (TRL, PEFT, Datasets)
├── config.py              # Centralized configuration (hyperparameters, naming)
├── run_training.sh        # SLURM job submission script
├── setup.sh               # venv creation and dependency installation
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (tokens, flags) 
├── .env.example           # Template for .env
├── models/                # Saved checkpoints (created at runtime)
└── logs/                  # SLURM output (.out) and error (.err) files
```

The venv and HuggingFace model cache live one level up, shared across runs:

```text
/mnt/storage_6/project_data/YOUR_GRANT/
├── venv/                  # Shared Python virtual environment
└── hf_cache/              # HuggingFace model cache (prevents redownloads)
```

---

## Troubleshooting

* **Job pending (`PD`):** Normal — waiting for a free GPU. Check status with `squeue -u $USER`.
* **`ERROR: Set GRANT in .env`:** The scripts can't find your project path. Make sure `.env` has `GRANT=your_grant_id` and that you run `sbatch`/`bash setup.sh` from the `supervised-fine-tuning` directory.
* **`OutOfMemoryError (OOM)`:**
  * *System RAM* (during model load): Increase `#SBATCH --mem` in `run_training.sh`.
  * *GPU VRAM* (during training): Reduce `TRAIN_BATCH_SIZE` in `config.py` or set `QUANTIZATION=4b` in `.env`.
* **`403 Forbidden` on HF Hub:** Your HF token is missing the **"Create and manage repos"** permission. Edit the token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) or set `PUSH_TO_HUB=false` in `.env` to disable hub pushing.
* **W&B Sync Errors:** Ensure compute nodes have outbound internet access or run `wandb offline` and sync manually later.
