import torch
import wandb
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from config import Config as cfg


class Trainer:
    def __init__(self):
        self._setup_auth()
        self.train_dataset, self.val_dataset = self._load_dataset()
        self.quant_config = self._build_quant_config()
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.lora_config = self._build_lora_config()
        self.sft_config = self._build_sft_config()

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
        raw_dataset = load_dataset(cfg.DATASET_NAME, split='train')
        dataset = raw_dataset.train_test_split(test_size=cfg.VAL_SIZE, seed=cfg.RANDOM_SEED)
        train = dataset['train']
        val = dataset['test']
        if cfg.MAX_TRAIN_SAMPLES > 0:
            train = train.select(range(min(cfg.MAX_TRAIN_SAMPLES, len(train))))
        return train, val

    @staticmethod
    def _build_quant_config():
        if cfg.QUANTIZATION == "4b":
            print("Applying 4-bit NormalFloat (NF4) quantization...")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if cfg.USE_BF16 else torch.float16,
                bnb_4bit_quant_type="nf4"
            )
        elif cfg.QUANTIZATION == "8b":
            print("Applying 8-bit quantization...")
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16 if cfg.USE_BF16 else torch.float16,
            )
        print("No quantization applied. Loading model in full precision (bf16/fp16).")
        return None

    @staticmethod
    def _load_tokenizer():
        tokenizer = AutoTokenizer.from_pretrained(cfg.BASE_MODEL, trust_remote_code=True)
        tokenizer.padding_side = "right"
        return tokenizer

    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            cfg.BASE_MODEL,
            device_map="auto",
            quantization_config=self.quant_config,
            dtype=torch.bfloat16,
            attn_implementation="sdpa"
        )
        # Add dedicated pad token to vocab and resize embeddings to train its weights
        self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(self.tokenizer))
        if self.quant_config is not None:
            model = prepare_model_for_kbit_training(model)
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
            modules_to_save=["embed_tokens", "lm_head"]
        )

    @staticmethod
    def _build_sft_config():
        return SFTConfig(
            output_dir=cfg.OUTPUT_DIR,
            num_train_epochs=cfg.EPOCHS,
            per_device_train_batch_size=cfg.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=cfg.EVAL_BATCH_SIZE,
            gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
            optim=cfg.OPTIMIZER,
            save_steps=cfg.SAVE_STEPS,
            save_total_limit=cfg.SAVE_LIMIT,
            logging_steps=cfg.LOG_STEPS,
            learning_rate=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
            fp16=not cfg.USE_BF16,
            bf16=cfg.USE_BF16,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=cfg.WARMUP_RATIO,
            lr_scheduler_type=cfg.LR_SCHEDULER_TYPE,
            report_to="wandb" if cfg.LOG_TO_WANDB else "none",
            run_name=cfg.RUN_NAME,
            max_length=cfg.MAX_SEQUENCE_LENGTH,
            save_strategy="steps",
            hub_strategy="every_save",
            push_to_hub=cfg.PUSH_TO_HUB,
            hub_model_id=cfg.HUB_MODEL_NAME,
            hub_token=cfg.HF_TOKEN,
            hub_private_repo=True,
            eval_strategy="steps",
            eval_steps=cfg.SAVE_STEPS,
            load_best_model_at_end=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            seed=cfg.RANDOM_SEED,
            data_seed=cfg.RANDOM_SEED
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

        print("Starting SFT training process...")
        fine_tuning.train()

        if cfg.PUSH_TO_HUB:
            print(f"Pushing trained model to HF Hub: {cfg.PROJECT_RUN_NAME}...")
            fine_tuning.model.push_to_hub(cfg.PROJECT_RUN_NAME, private=True)

        if cfg.LOG_TO_WANDB:
            wandb.finish()


if __name__ == "__main__":
    assert cfg.QUANTIZATION in ("none", "4b", "8b"), f"Unknown QUANTIZATION: {cfg.QUANTIZATION}"
    set_seed(cfg.RANDOM_SEED)
    Trainer().run()
