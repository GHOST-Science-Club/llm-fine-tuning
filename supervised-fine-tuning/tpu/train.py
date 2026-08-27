"""
TPU variant of the SFT training script (see ../gpu/train.py for the original).

All deviations from the GPU version are marked with '#changed', in the same
spirit as the "TRL-on-TPU-working" showcase notebook this port is based on
(see tpu/README.md for the link).
"""
import os

import torch
# changed: torch_xla is the PyTorch TPU backend (there is no CUDA on a TPU VM).
# Importing it registers the XLA device with PyTorch. The monkeypatch below is
# carried over from the showcase notebook, where it was found necessary because
# some transformers code probes `torch.xla` directly.
import torch_xla
import torch_xla.core.xla_model as xm  # noqa: F401  (import triggers registration)
import torch_xla.runtime as xr

torch.xla = torch_xla

import wandb
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
# changed: no BitsAndBytesConfig / prepare_model_for_kbit_training imports —
# bitsandbytes is CUDA-only, so the TPU variant trains a bf16 LoRA instead of QLoRA
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from config import cfg


class Trainer:
    def __init__(self):
        self._setup_xla()
        self._setup_auth()
        self.train_dataset, self.val_dataset = self._load_dataset()
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.lora_config = self._build_lora_config()
        self.sft_config = self._build_sft_config()

    @staticmethod
    def _setup_xla():
        # changed: persistent compilation cache. XLA compiles the whole training
        # step into a fixed program on first use — for an 11B model this can take
        # many minutes and looks like a hang at step 0. The cache stores the
        # compiled program on disk so every later run starts immediately.
        # The cache is an optimisation, never a requirement — nothing here may take
        # down a multi-hour run. Catch broadly: an unwritable ~/.cache (OSError),
        # a renamed API (AttributeError/TypeError) and XLA runtime errors are all
        # equally survivable. KeyboardInterrupt/SystemExit derive from
        # BaseException, so Ctrl+C still works.
        try:
            xr.initialize_cache(os.path.expanduser("~/.cache/xla_compile"), readonly=False)
        except Exception as e:
            print(f"Warning: could not enable the XLA compilation cache: {e}")

        # changed: SPMD mode is what lets FSDPv2 shard the model across all TPU
        # chips. Must be enabled before the model is created/moved.
        if cfg.USE_FSDP_V2:
            xr.use_spmd()

    @staticmethod
    def _setup_auth():
        if cfg.HF_TOKEN:
            login(cfg.HF_TOKEN, add_to_git_credential=True)
        else:
            print("Warning: HF_TOKEN not found. Model pushing will fail.")

        if cfg.LOG_TO_WANDB and cfg.WANDB_API_KEY:
            wandb.login(key=cfg.WANDB_API_KEY)
            wandb.init(project=cfg.PROJECT_NAME, name=cfg.RUN_NAME)

    @staticmethod
    def _load_dataset():
        print(f"Loading dataset: {cfg.DATASET_NAME}...")
        try:
            raw_dataset = load_dataset(cfg.DATASET_NAME, split='train')
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset '{cfg.DATASET_NAME}'. Check DATASET_NAME in config.") from e
        dataset = raw_dataset.train_test_split(test_size=cfg.VAL_SIZE, seed=cfg.RANDOM_SEED)
        train = dataset['train']
        val = dataset['test']
        if cfg.MAX_TRAIN_SAMPLES > 0:
            train = train.select(range(min(cfg.MAX_TRAIN_SAMPLES, len(train))))
        return train, val

    @staticmethod
    def _load_tokenizer():
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg.BASE_MODEL, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer for '{cfg.BASE_MODEL}'. Check BASE_MODEL and HF_TOKEN in .env.") from e
        tokenizer.padding_side = "right"
        # changed: the GPU version adds a brand-new <pad> token, resizes the
        # embedding matrix and then has to train embed_tokens + lm_head. On TPU
        # we reuse the existing <unk> token as padding instead: no resize, no
        # extra ~0.5 GB of trainable parameters + optimizer state, and the
        # vocabulary keeps a static size (XLA-friendly). With packing enabled
        # (see _build_sft_config) padding is barely used anyway.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
        return tokenizer

    def _load_model(self):
        try:
            # changed: no quantization_config (bitsandbytes is CUDA-only) — the
            # model is loaded in bfloat16, the TPU's native training dtype.
            # changed: no device_map="auto" (that is accelerate's CUDA dispatch).
            # The model stays on CPU here; the HF Trainer moves it to the XLA
            # device — or, with FSDPv2, shards it across chips. Moving it manually
            # with .to(xla_device) like the showcase did would materialize all
            # 11B parameters on ONE chip and OOM before sharding could happen.
            # changed: attn_implementation "eager" instead of "sdpa" — the fused
            # SDPA/flash kernels target CUDA; eager attention lowers cleanly
            # through the XLA compiler.
            model = AutoModelForCausalLM.from_pretrained(
                cfg.BASE_MODEL,
                dtype=torch.bfloat16,
                attn_implementation="eager",
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{cfg.BASE_MODEL}'. Check BASE_MODEL and HF_TOKEN in .env.") from e
        # changed: no add_special_tokens / resize_token_embeddings (see
        # _load_tokenizer) and no prepare_model_for_kbit_training (no quantization)
        return model

    @staticmethod
    def _build_lora_config():
        return LoraConfig(
            lora_alpha=cfg.LORA_ALPHA,
            lora_dropout=cfg.LORA_DROPOUT,
            r=cfg.LORA_R,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=cfg.TARGET_MODULES,
            # changed: no modules_to_save=["embed_tokens", "lm_head"] — we did
            # not resize the vocabulary, so the (very large) embedding matrices
            # can stay frozen like the rest of the base model
        )

    @staticmethod
    def _build_sft_config():
        # changed: FSDPv2 via XLA SPMD — shards parameters, gradients and
        # optimizer states across all TPU chips so an 11B model fits. This is
        # the officially supported transformers+TPU sharding path.
        fsdp_kwargs = {}
        if cfg.USE_FSDP_V2:
            fsdp_kwargs = dict(
                fsdp="full_shard",
                fsdp_config={
                    "transformer_layer_cls_to_wrap": [cfg.FSDP_LAYER_CLS],
                    "xla": True,
                    "xla_fsdp_v2": True,
                    # FSDPv2's own gradient checkpointing (replaces the classic
                    # gradient_checkpointing flag below)
                    "xla_fsdp_grad_ckpt": True,
                },
            )
        return SFTConfig(
            output_dir=cfg.OUTPUT_DIR,
            num_train_epochs=cfg.EPOCHS,
            per_device_train_batch_size=cfg.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=cfg.EVAL_BATCH_SIZE,
            gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
            # changed: adamw_torch — the paged_adamw_* optimizers of the GPU
            # version come from bitsandbytes (CUDA-only)
            optim=cfg.OPTIMIZER,
            save_steps=cfg.SAVE_STEPS,
            save_total_limit=cfg.SAVE_LIMIT,
            logging_steps=cfg.LOG_STEPS,
            learning_rate=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
            # changed: bf16 forced on — the GPU version keys this off CUDA
            # compute capability, which is meaningless on a TPU; TPUs are
            # natively bfloat16 (same reasoning as the showcase's bf16=True)
            fp16=False,
            bf16=True,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=cfg.WARMUP_RATIO,
            lr_scheduler_type=cfg.LR_SCHEDULER_TYPE,
            report_to="wandb" if cfg.LOG_TO_WANDB else "none",
            run_name=cfg.RUN_NAME,
            max_length=cfg.MAX_SEQUENCE_LENGTH,
            # changed: packing with the 'wrapped' strategy + drop_last. XLA
            # compiles one program per unique tensor shape — any new shape
            # triggers a full recompilation that takes minutes. Packing makes
            # every batch exactly (batch_size, max_length); drop_last removes
            # the one ragged final batch. This is the single most important
            # TPU performance setting.
            packing=True,
            packing_strategy="wrapped",
            dataloader_drop_last=True,
            save_strategy="steps",
            hub_strategy="every_save",
            push_to_hub=cfg.PUSH_TO_HUB,
            hub_model_id=cfg.HUB_MODEL_NAME,
            hub_token=cfg.HF_TOKEN,
            hub_private_repo=True,
            eval_strategy="steps",
            eval_steps=cfg.SAVE_STEPS,
            load_best_model_at_end=True,
            # changed: classic gradient checkpointing only outside FSDPv2 —
            # inside it, xla_fsdp_grad_ckpt (above) does the same job the XLA way
            gradient_checkpointing=not cfg.USE_FSDP_V2,
            gradient_checkpointing_kwargs={"use_reentrant": False} if not cfg.USE_FSDP_V2 else None,
            seed=cfg.RANDOM_SEED,
            data_seed=cfg.RANDOM_SEED,
            **fsdp_kwargs,
        )

    def _formatting_prompts_func(self, example: dict[str, str]) -> str:
        messages = [
            {"role": "user", "content": example["query"]},
            {"role": "assistant", "content": example["response"]},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def run(self):
        fine_tuning = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            peft_config=self.lora_config,
            args=self.sft_config,
            formatting_func=self._formatting_prompts_func
        )

        print("Starting SFT training process on TPU...")
        print("(the first optimization step compiles the XLA graph — a long pause there is normal)")
        try:
            fine_tuning.train()
        # ConnectionError/TimeoutError are OSError subclasses, so they must be caught
        # first — hub_strategy="every_save" uploads mid-training, and a failed upload
        # would otherwise be reported as a disk problem.
        except (ConnectionError, TimeoutError) as e:
            raise RuntimeError("Training interrupted by a network error (Hub upload?). Check connectivity and HF_TOKEN.") from e
        except OSError as e:
            raise RuntimeError("Training interrupted due to a disk error. Check available disk space and OUTPUT_DIR permissions.") from e

        if cfg.PUSH_TO_HUB:
            # cfg.HUB_MODEL_NAME, not PROJECT_RUN_NAME: an unqualified name resolves to
            # the token owner's namespace, which is the wrong repo whenever HF_USER is
            # an org. Must match hub_model_id in _build_sft_config().
            print(f"Pushing trained model to HF Hub: {cfg.HUB_MODEL_NAME}...")
            fine_tuning.model.push_to_hub(cfg.HUB_MODEL_NAME, private=True)

        if cfg.LOG_TO_WANDB:
            wandb.finish()


if __name__ == "__main__":
    set_seed(cfg.RANDOM_SEED)
    Trainer().run()
