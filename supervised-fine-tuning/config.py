import os
from pathlib import Path
from datetime import datetime
import torch
from dotenv import load_dotenv
from dataclasses import dataclass


# Load .env file automatically upon import
ROOT_PATH = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_PATH / ".env", override=True)

@dataclass
class Config:

    """
    Configuration class that centralizes all settings for the fine-tuning process.
    Values are loaded from environment variables with sensible defaults where applicable.
    """

    # --- Environment Variables ---
    PUSH_TO_HUB = os.getenv('PUSH_TO_HUB', 'false').lower() == 'true'
    LOG_TO_WANDB = os.getenv('LOG_TO_WANDB', 'false').lower() == 'true'
    HF_TOKEN = os.getenv('HF_TOKEN', '')
    WANDB_API_KEY = os.getenv('WANDB_API_KEY', '')

    # Defaults to 'none' for H100, can be overridden to '4b' or '8b' via .env
    QUANTIZATION = os.getenv('QUANTIZATION', 'none').lower()

    # Set fixed seed to make experiments reproducible
    RANDOM_SEED = 42

    # --- Model and Project Configuration ---
    BASE_MODEL = "speakleash/Bielik-11B-v3.0-Instruct"
    PROJECT_NAME = "bielik-tuning"
    HF_USER = "erybie222"
    DATASET_NAME = "meta-math/MetaMathQA"

    # --- Dynamic Naming ---
    RUN_NAME = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
    PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
    HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

    # --- Hyperparameters: Overall ---
    EPOCHS = 3
    TRAIN_BATCH_SIZE = 4
    EVAL_BATCH_SIZE = 4
    MAX_SEQUENCE_LENGTH = 4096
    GRADIENT_ACCUMULATION_STEPS = 4

    # --- Hyperparameters: LoRA ---
    LORA_R = 32
    LORA_ALPHA = LORA_R * 2
    TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    LORA_DROPOUT = 0.1

    # --- Hyperparameters: Training ---
    LEARNING_RATE = 1e-4
    WARMUP_RATIO = 0.05
    LR_SCHEDULER_TYPE = 'cosine'
    WEIGHT_DECAY = 0.001

    if QUANTIZATION == '4b':
        OPTIMIZER = "paged_adamw_4bit"
    if QUANTIZATION == '8b':
        OPTIMIZER = "paged_adamw_8bit"
    else:
        OPTIMIZER = "paged_adamw_32bit"

    # --- Tracking & Validation ---
    VAL_SIZE = 1000  # Number of samples to hold out for the validation split
    SAVE_LIMIT = 10  # Maximum number of checkpoints to keep
    SAVE_STEPS = 200  # Save and evaluate every 200 steps
    LOG_STEPS = 10  # Log metrics to W&B every 10 steps
    MAX_TRAIN_SAMPLES = int(os.getenv('MAX_TRAIN_SAMPLES', '0')) # Limit training examples from dataset
    save_total_limit = 10,

    # --- Hardware Capabilities ---
    _capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    USE_BF16 = _capability[0] >= 8  # Automatically use bfloat16 for Ampere architecture or newer (like H100)