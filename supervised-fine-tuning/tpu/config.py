import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from dataclasses import dataclass, field


# Load .env file automatically upon import
MODULE_PATH = Path(__file__).resolve().parent
load_dotenv(MODULE_PATH / ".env", override=True)


@dataclass
class Config:
    """
    Configuration class that centralizes all settings for the fine-tuning process on TPU.
    Values are loaded from environment variables with sensible defaults where applicable.

    #changed vs the GPU config:
      - no QUANTIZATION option — bitsandbytes (QLoRA) is CUDA-only; the TPU
        variant trains a plain bfloat16 LoRA instead
      - OPTIMIZER fixed to adamw_torch — the paged_adamw_* optimizers also
        come from bitsandbytes
      - no CUDA capability detection — TPUs are natively bfloat16, bf16 is
        always on (this file no longer needs to import torch at all)
      - BASE_MODEL / MAX_SEQUENCE_LENGTH / TRAIN_BATCH_SIZE overridable from
        .env so the same code can run cheap smoke tests (see tpu/README)
      - new TPU-specific flags: USE_FSDP_V2 and FSDP_LAYER_CLS control SPMD
        sharding of the model across TPU chips
    """

    # --- Environment Variables ---
    PUSH_TO_HUB: bool = os.getenv('PUSH_TO_HUB', 'false').lower() == 'true'
    LOG_TO_WANDB: bool = os.getenv('LOG_TO_WANDB', 'false').lower() == 'true'
    HF_TOKEN: str = field(default_factory=lambda: os.getenv('HF_TOKEN', ''))
    WANDB_API_KEY: str = field(default_factory=lambda: os.getenv('WANDB_API_KEY', ''))
    MAX_TRAIN_SAMPLES: int = field(default_factory=lambda: int(os.getenv('MAX_TRAIN_SAMPLES', '0')))

    # Set fixed seed to make experiments reproducible
    RANDOM_SEED: int = 42

    # --- Model and Project Configuration ---
    # BASE_MODEL is env-overridable so smoke tests can swap in a tiny model
    BASE_MODEL: str = field(default_factory=lambda: os.getenv('BASE_MODEL', "speakleash/Bielik-11B-v3.0-Instruct"))
    PROJECT_NAME: str = "bielik-tuning-tpu"
    HF_USER: str = field(default_factory=lambda: os.getenv('HF_USER', ''))
    # NOTE: For testing purposess, hardcoded to MetaMathQA. In practice, this should be set to our own dataset.
    # The dataset should have columns: "query" and "response"(enforced in _formatting_prompts_func in train.py).
    DATASET_NAME: str = "meta-math/MetaMathQA"

    # --- TPU-specific ---
    # FSDPv2 (SPMD) shards model weights, gradients and optimizer states across
    # all TPU chips. Required for Bielik-11B (~22 GB of bf16 weights probably won't fit
    # a single chip together with activations). Disable only for smoke tests
    # with small models on a single chip or on CPU.
    USE_FSDP_V2: bool = os.getenv('USE_FSDP_V2', 'true').lower() == 'true'
    # Decoder-layer class FSDP wraps/shards on. Bielik is Mistral-based.
    # For smoke-test models use the matching class, e.g. Qwen2DecoderLayer.
    FSDP_LAYER_CLS: str = field(default_factory=lambda: os.getenv('FSDP_LAYER_CLS', 'MistralDecoderLayer'))

    # --- Dynamic Naming (computed in __post_init__) ---
    RUN_NAME: str = field(default_factory=lambda: f"{datetime.now():%Y-%m-%d_%H.%M.%S}")
    PROJECT_RUN_NAME: str = field(init=False, default='')
    HUB_MODEL_NAME: str = field(init=False, default='')
    OUTPUT_DIR: str = field(init=False, default='')

    # --- Hyperparameters: Overall ---
    EPOCHS: int = 3
    # NOTE: under SPMD the whole TPU acts as one device, so this is the GLOBAL
    # batch size (it gets sharded across chips), not a per-chip one.
    TRAIN_BATCH_SIZE: int = field(default_factory=lambda: int(os.getenv('TRAIN_BATCH_SIZE', '4')))
    EVAL_BATCH_SIZE: int = field(default_factory=lambda: int(os.getenv('EVAL_BATCH_SIZE', '4')))
    MAX_SEQUENCE_LENGTH: int = field(default_factory=lambda: int(os.getenv('MAX_SEQUENCE_LENGTH', '4096')))
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
    # changed: fixed — the GPU version picks a paged_adamw_* variant based on
    # quantization, but those optimizers live in bitsandbytes (CUDA-only)
    OPTIMIZER: str = 'adamw_torch'

    # --- Tracking & Validation ---
    # SAVE_STEPS/LOG_STEPS are env-overridable: a smoke test's total step count
    # (few dozen samples) can be smaller than the GPU-tuned default of 200,
    # which would mean no checkpoint is ever saved — fatal when
    # load_best_model_at_end=True tries to load one that doesn't exist.
    VAL_SIZE: int = field(default_factory=lambda: int(os.getenv('VAL_SIZE', '1000')))
    SAVE_LIMIT: int = 10
    SAVE_STEPS: int = field(default_factory=lambda: int(os.getenv('SAVE_STEPS', '200')))
    LOG_STEPS: int = field(default_factory=lambda: int(os.getenv('LOG_STEPS', '10')))

    def __post_init__(self):
        self.LORA_ALPHA = self.LORA_R * 2

        self.PROJECT_RUN_NAME = f"{self.PROJECT_NAME}-{self.RUN_NAME}"
        if self.PUSH_TO_HUB and not self.HF_USER:
            raise ValueError("PUSH_TO_HUB=true but HF_USER is not set. Set HF_USER in .env to your HF username.")
        # Fail here rather than after hours of training: without a token the push at
        # the very end is the first thing that notices, and hub_token="" is sent as an
        # empty bearer instead of falling back to the cached login().
        if self.PUSH_TO_HUB and not self.HF_TOKEN:
            raise ValueError("PUSH_TO_HUB=true but HF_TOKEN is not set. Set HF_TOKEN in .env.")
        self.HUB_MODEL_NAME = f"{self.HF_USER}/{self.PROJECT_RUN_NAME}"
        self.OUTPUT_DIR = f"models/{self.PROJECT_RUN_NAME}"


cfg = Config()
