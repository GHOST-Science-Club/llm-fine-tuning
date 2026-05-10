# Bielik Evaluation — HPC Eagle

Runs the Polish math benchmark on **Bielik-11B-v3.0-Instruct** using `lm_eval` on a single H100 GPU.

---

## Setup (one-time per user)

SSH into the cluster and add your project path to `~/.bashrc`:

```bash
echo 'export PROJECT_PATH="/mnt/storage_6/project_data/YOUR_GRANT"' >> ~/.bashrc
source ~/.bashrc
```

Replace `YOUR_GRANT` with your actual grant ID (e.g. `pl0966-02`).

## HuggingFace Authentication (one-time per user)

The Bielik model is gated — you need a HuggingFace account with access granted.

1. Request access at: https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct
2. Generate a read token at: https://huggingface.co/settings/tokens
3. On the cluster, activate the venv and log in:

```bash
source $PROJECT_PATH/venv/bin/activate
export HF_HOME="$PROJECT_PATH/hf_cache"
hf auth login
```

Token will be saved to `$PROJECT_PATH/hf_cache/token` and reused automatically by all subsequent jobs.

---

## Running the Evaluation

Navigate to the evaluation directory and submit the job:

```bash
cd $PROJECT_PATH/llm-fine-tuning/evaluation
sbatch run_benchmark.sh
```

On first run, the script automatically creates a virtual environment and installs all dependencies. Subsequent runs reuse the existing venv and cached model.

---

## Monitoring

```bash
# Check job status
squeue -u $USER

# Follow logs in real time (replace JOBID with your job number)
tail -f logs_JOBID.out

# Check errors
cat logs_JOBID.err
```

---

## Project Structure

```
$PROJECT_PATH/
├── llm-fine-tuning/
│   └── evaluation/
│       ├── run_benchmark.sh       # SLURM job script — submit this
│       ├── run_eval.py            # Evaluation entry point
│       ├── benchmark.yaml         # lm_eval task config
│       ├── benchmark.jsonl        # Math benchmark dataset
│       └── math_verify_metric.py  # Custom LaTeX answer metric
├── hf_cache/                      # HuggingFace model cache
├── pip_cache/                     # pip cache
└── venv/                          # Python virtual environment
```

---

## Copying Files to the Cluster

From your local machine:

```bash
rsync -avz --progress \
  -e "ssh -i ~/.ssh/id_ed25519" \
  ./evaluation/ \
  youruser@eagle.man.poznan.pl:$PROJECT_PATH/llm-fine-tuning/evaluation/
```

---

## Troubleshooting

**Job pending (`PD`)** — normal, waiting for a free GPU. Check with `squeue -u $USER`.

**`ModuleNotFoundError: No module named 'lm_eval'`** — venv is broken, delete and resubmit:
```bash
rm -rf $PROJECT_PATH/venv
sbatch run_benchmark.sh
```

**`Disk quota exceeded`** — check usage:
```bash
du -sh $PROJECT_PATH/*
```