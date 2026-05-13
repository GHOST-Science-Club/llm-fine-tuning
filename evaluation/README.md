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
3. Create a `.env` file in the `evaluation` directory and paste your token.
4. You can do it manually or use command given below.

```bash
echo 'HF_TOKEN="your_hf_token_here"' > $PROJECT_PATH/evaluation/.env
```

Token will be saved to `$PROJECT_PATH/evaluation/.env` and reused automatically by all subsequent jobs.

---

## Copying Files to the Cluster

From your local machine:

```bash
cd /mnt/e/Projects/Pycharm/llm-fine-tuning/evaluation  # (Your local project directory)
scp -i ~/.ssh/id_YOUR_ID -r ./* username@eagle.man.poznan.pl:/mnt/storage_6/project_data/YOUR_GRANT/evaluation```
```

---

## Running the Setup

If you want to run that evaluation for first time, you have to install dependencies.
Navigate to the evaluation directory and run setup.sh script, which installs those dependencies and creates virtual environment for you.

```bash
cd $PROJECT_PATH/evaluation
bash setup.sh
```
---

## Running the Evaluation

Navigate to the evaluation directory and submit the job:

```bash
cd $PROJECT_PATH/evaluation
sbatch run_benchmark.sh
```


---

## Monitoring

```bash
# Check job status
squeue -u $USER

# Follow logs in real time (replace JOBID with your job number)
tail -f logs/JOBID.out

# Check errors
cat logs/JOBID.err
```

---

## Project Structure

```
$PROJECT_PATH/
├── evaluation/
│       ├── run_benchmark.sh       # SLURM job script
│       ├── setup.sh               # Script for installing dependencies and creating virtual environment
│       ├── run_eval.py            # Evaluation entry point
│       ├── benchmark.yaml         # lm_eval task config
│       ├── benchmark.jsonl        # Math benchmark dataset
│       ├── math_verify_metric.py  # Custom LaTeX answer metric
│       └── requirements.txt       # List of dependencies
├── hf_cache/                      # HuggingFace model cache
├── pip_cache/                     # pip cache
└── venv/                          # Python virtual environment
```

---



## Troubleshooting

**Job pending (`PD`)** — normal, waiting for a free GPU. Check with `squeue -u $USER`.

**`ModuleNotFoundError: No module named 'lm_eval'`** — venv is broken, delete it and reinstall:
```bash
rm -rf $PROJECT_PATH/venv
cd $PROJECT_PATH/evaluation
bash setup.sh
sbatch run_benchmark.sh
```

**`Disk quota exceeded`** — check usage:
```bash
du -sh $PROJECT_PATH/*
```