import os
from pathlib import Path
from datetime import datetime
import torch
from dotenv import load_dotenv
from dataclasses import dataclass, field


# Load .env file automatically upon import
MODULE_PATH = Path(__file__).resolve().parent
load_dotenv(MODULE_PATH / ".env", override=True)


@dataclass
class Config:
    """
    Configuration class that centralizes all settings for the fine-tuning process.
    Values are loaded from environment variables with sensible defaults where applicable.
    """

    # --- Environment Variables ---
    PUSH_TO_HUB: bool = os.getenv('PUSH_TO_HUB', 'false').lower() == 'true'
    LOG_TO_WANDB: bool = os.getenv('LOG_TO_WANDB', 'false').lower() == 'true'
    HF_TOKEN: str = field(default_factory=lambda: os.getenv('HF_TOKEN', ''))
    WANDB_API_KEY: str = field(default_factory=lambda: os.getenv('WANDB_API_KEY', ''))
    QUANTIZATION: str = field(default_factory=lambda: os.getenv('QUANTIZATION', 'none').lower())
    MAX_TRAIN_SAMPLES: int = field(default_factory=lambda: int(os.getenv('MAX_TRAIN_SAMPLES', '0')))

    # Set fixed seed to make experiments reproducible
    RANDOM_SEED: int = 42

    # --- Model and Project Configuration ---
    BASE_MODEL: str = "speakleash/Bielik-11B-v3.0-Instruct"
    PROJECT_NAME: str = "bielik-tuning"
    HF_USER: str = "erybie222"
    DATASET_NAME: str = "meta-math/MetaMathQA"

    # --- Dynamic Naming (computed in __post_init__) ---
    RUN_NAME: str = field(default_factory=lambda: f"{datetime.now():%Y-%m-%d_%H.%M.%S}")
    PROJECT_RUN_NAME: str = field(init=False, default='')
    HUB_MODEL_NAME: str = field(init=False, default='')
    OUTPUT_DIR: str = field(init=False, default='')

    # --- Hyperparameters: Overall ---
    EPOCHS: int = 3
    TRAIN_BATCH_SIZE: int = 4
    EVAL_BATCH_SIZE: int = 4
    MAX_SEQUENCE_LENGTH: int = 4096
    GRADIENT_ACCUMULATION_STEPS: int = 4

    # --- Hyperparameters: LoRA ---
    LORA_R: int = 32
    LORA_ALPHA: int = field(init=False, default=0)  # computed from LORA_R in __post_init__
    TARGET_MODULES: list = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    LORA_DROPOUT: float = 0.1

    # --- Hyperparameters: Training ---
    LEARNING_RATE: float = 1e-4
    WARMUP_RATIO: float = 0.05
    LR_SCHEDULER_TYPE: str = 'cosine'
    WEIGHT_DECAY: float = 0.001
    OPTIMIZER: str = field(init=False, default='')  # computed from QUANTIZATION in __post_init__

    # --- Tracking & Validation ---
    VAL_SIZE: int = 1000
    SAVE_LIMIT: int = 10
    SAVE_STEPS: int = 200
    LOG_STEPS: int = 10

    # --- Hardware Capabilities (computed in __post_init__) ---
    _CAPABILITY: tuple = field(init=False, default_factory=lambda: (0, 0))
    USE_BF16: bool = field(init=False, default=False)

    def __post_init__(self):
        self.LORA_ALPHA = self.LORA_R * 2

        self.PROJECT_RUN_NAME = f"{self.PROJECT_NAME}-{self.RUN_NAME}"
        self.HUB_MODEL_NAME = f"{self.HF_USER}/{self.PROJECT_RUN_NAME}"
        self.OUTPUT_DIR = f"models/{self.PROJECT_RUN_NAME}"

        if self.QUANTIZATION == '4b':
            self.OPTIMIZER = "paged_adamw_4bit"
        elif self.QUANTIZATION == '8b':
            self.OPTIMIZER = "paged_adamw_8bit"
        else:
            self.OPTIMIZER = "paged_adamw_32bit"

        if torch.cuda.is_available():
            self._CAPABILITY = torch.cuda.get_device_capability()
            self.USE_BF16 = self._CAPABILITY[0] >= 8
        else:
            print("Warning: No CUDA-compatible GPU detected. Training will be performed on CPU, which may be very slow.")


cfg = Config()
