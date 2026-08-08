# Bielik Supervised Fine-Tuning — TPU (Google Cloud, PyTorch/XLA)

TPU port of the GPU pipeline in [`../gpu/`](../gpu/README.md). Fine-tunes **Bielik-11B-v3.0-Instruct** with LoRA + `trl`'s `SFTTrainer` on Google Cloud TPUs, using the changes validated in the [trl_on_tpu_working.ipynb](https://colab.research.google.com/drive/1vcHi8_GcEtgDvJvQxLE87HmBODmV308B) Colab showcase.

All deviations from the GPU version are marked with `#changed` comments directly in the code.

---

## Setup (one-time per user)

1. **HuggingFace:** Bielik is gated — request access at [speakleash/Bielik-11B-v3.0-Instruct](https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct) and generate a **Write** token (+ "Create and manage repos" if pushing checkpoints) at your [settings page](https://huggingface.co/settings/tokens).
2. **Weights & Biases** (optional): account at [wandb.ai](https://wandb.ai), API key from your [authorizations page](https://wandb.ai/authorize).
3. `cp .env.example .env` and fill in values.

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | yes | HuggingFace token (see above) |
| `HF_USER` | yes (if `PUSH_TO_HUB=true`) | Your HF username — no default, fails fast at startup if unset while pushing |
| `WANDB_API_KEY` | yes (if logging) | W&B API key |
| `PUSH_TO_HUB` / `LOG_TO_WANDB` | no | `true` to enable each |
| `MAX_TRAIN_SAMPLES` | no | Limit training examples; `0` = full dataset |
| `USE_FSDP_V2` / `FSDP_LAYER_CLS` | no | SPMD sharding across TPU chips — see below |

---

## Monitoring

No SLURM/`squeue` here — a TPU VM runs your script directly, so progress prints straight to the terminal (and to `logs/train_<timestamp>.log` via `run_training.sh`'s `tee`). If `LOG_TO_WANDB=true`, loss/eval curves also appear on your [W&B dashboard](https://wandb.ai) in real time.

---

## Running on a real TPU (the grant) 
for now this part was written by claude and not validated as the grant in not opened yet. `setup.sh`/`run_training.sh` themselves haven't been run yet either (only their equivalent manual commands were, on CPU and Colab) — should work the same, just not yet confirmed.

```bash
# 1. Create a TPU VM (v4-8 or v5e-8 recommended; adjust zone/project to the grant)
gcloud compute tpus tpu-vm create bielik-sft \
    --zone=us-central2-b --accelerator-type=v4-8 --version=tpu-ubuntu2204-base

# 2. Copy this folder over and SSH in
gcloud compute tpus tpu-vm scp --recurse ./tpu bielik-sft:~/tpu --zone=us-central2-b
gcloud compute tpus tpu-vm ssh bielik-sft --zone=us-central2-b

# 3. On the TPU VM:
cd ~/tpu
bash setup.sh
cp .env.example .env   # then fill in HF_TOKEN etc.
bash run_training.sh
```

Notes:
* **Under SPMD, `TRAIN_BATCH_SIZE` is the global batch size** (the whole TPU acts as one device and the batch is sharded across chips).
* **A long pause at step 0 is normal** — that's XLA compiling the training program. It's cached in `~/.cache/xla_compile` for subsequent runs.
* Checkpoints land in `models/bielik-tuning-tpu-<timestamp>/` (LoRA adapter only — small).

---

## Testing without the grant

Yes — there are two free tiers, from cheapest to most realistic:

### 1. Your own laptop: XLA on CPU (no TPU at all)

torch_xla can run on a plain CPU. It exercises the *exact same code path and XLA compiler* — it just executes the compiled program on CPU. This catches ~all porting bugs (wrong API, shape problems, CUDA-only imports) for free. It says nothing about speed — verified end-to-end below, a single training step of the tiny smoke model took **~5 minutes on CPU** (that's the one-time XLA compile; real TPU hardware does this in seconds).

```bash
cd tpu
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip

# torch_xla WITHOUT the [tpu] extra, no libtpu index — this is the CPU-only build.
# As of writing PyPI has no 2.7.x wheels for Python 3.12; pin a matched pair instead
# (check `pip index versions torch_xla` if this drifts):
pip install torch_xla~=2.8.0
pip install torch==2.8.0   # must match the torch_xla version exactly

# the rest of the stack (skip requirements.txt — it's pinned for the [tpu] extra)
pip install "transformers<5.0.0" peft "trl>=0.23.1" datasets huggingface_hub \
    wandb python-dotenv accelerate sentencepiece protobuf

cp .env.example .env
# then uncomment the smoke-test block in .env (tiny model, FSDP off, short
# sequences) AND add these three — without them the default SAVE_STEPS=200
# exceeds a smoke test's total step count, so no checkpoint is ever saved and
# load_best_model_at_end crashes at the very end:
#   SAVE_STEPS=2
#   LOG_STEPS=1
#   VAL_SIZE=8

PJRT_DEVICE=CPU bash run_training.sh
```

Verified (Aug 2026, Qwen2.5-0.5B smoke config, via equivalent manual commands rather than `run_training.sh` itself): model load → LoRA wrap → packing → forward/backward → optimizer step → eval → checkpoint save all ran correctly with sane loss/accuracy numbers and a real `adapter_model.safetensors` on disk. The CPU run was interrupted before 100% (session teardown); full run-to-completion was confirmed on real TPU (section 2). One CPU-only quirk: repeated `Failed to deserialize executable: UNIMPLEMENTED` warnings — the compile cache only works on real TPU, so CPU always recompiles. Harmless.

### 2. Google Colab: free/cheap TPU runtime

the tpu training algorithm was tested on the colab.

the instructions on how to run it there are on [our google drive](https://drive.google.com/drive/folders/1LURnkjoqlK9dZey4IUwBisgjNTDBNmas)

Recommended smoke-test `.env` (works on both — this is the exact config verified in section 1 above):

```bash
BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
FSDP_LAYER_CLS=Qwen2DecoderLayer
USE_FSDP_V2=false
MAX_SEQUENCE_LENGTH=256
TRAIN_BATCH_SIZE=2
MAX_TRAIN_SAMPLES=32
SAVE_STEPS=2
LOG_STEPS=1
VAL_SIZE=8
```

---

## Troubleshooting (TPU-specific)

* **"It hangs at step 0"** — XLA is compiling. Wait it out once; the compile cache makes later runs fast.
* **Every step is slow / it keeps "hanging"** — a dynamic shape is leaking and forcing recompilation each step. Check that `packing=True` and `dataloader_drop_last=True` didn't get removed.
* **OOM on a single chip** — set `USE_FSDP_V2=true` (sharding), or lower `TRAIN_BATCH_SIZE` / `MAX_SEQUENCE_LENGTH`.
* **`ValueError` about the layer class** — `FSDP_LAYER_CLS` must match the model architecture: `MistralDecoderLayer` for Bielik, `Qwen2DecoderLayer` for the Qwen smoke model, `LlamaDecoderLayer` for Llama-family models.
* **Crash inside `generate()`** — re-add `model.config.sliding_window = None` (see the showcase's notes on SlidingWindowCache).

---

## Project Structure

```text
supervised-fine-tuning/tpu/
├── train.py               # TPU SFT script (#changed markers vs ../gpu/train.py)
├── config.py              # TPU config (no quantization, FSDP flags, env overrides)
├── run_training.sh        # direct launch on the TPU VM (no SLURM)
├── setup.sh               # venv + dependency installation
├── requirements.txt       # torch + torch_xla[tpu]; no bitsandbytes
├── .env.example           # template incl. smoke-test overrides
├── models/                # saved adapters (created at runtime)
├── logs/                  # training logs (created at runtime)
└── hf_cache/              # HF model cache (created at runtime)
```
