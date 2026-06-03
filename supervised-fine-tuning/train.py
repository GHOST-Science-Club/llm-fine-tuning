import torch
import wandb
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from config import Config as cfg


def formatting_prompts_func(example):
    """ Formats raw dataset example into ChatML template """
    messages = [
        {"role": "user",    "content": example["query"]},
        {"role": "assistant","content": example["response"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

if __name__ == "__main__":

    # 1. Authentication and Logging setup
    if cfg.HF_TOKEN:
        login(cfg.HF_TOKEN, add_to_git_credential=True)
    else:
        print("Warning: HF_TOKEN not found. Model pushing will fail.")

    if cfg.LOG_TO_WANDB and cfg.WANDB_API_KEY:
        wandb.login(key=cfg.WANDB_API_KEY)
        wandb.init(project=cfg.PROJECT_NAME, name=cfg.RUN_NAME)

    # 2. Dataset Loading and Splitting
    # MetaMathQA only provides a 'train' split. We load it and carve out a validation set.
    print(f"Loading dataset: {cfg.DATASET_NAME}...")
    raw_dataset = load_dataset(cfg.DATASET_NAME, split='train')
    dataset = raw_dataset.train_test_split(test_size=cfg.VAL_SIZE)
    train = dataset['train']
    val = dataset['test']

    # Limit train samples
    if cfg.MAX_TRAIN_SAMPLES > 0:
        actual_samples = min(cfg.MAX_TRAIN_SAMPLES, len(train))
        train = train.select(range(actual_samples))

    # 3. Quantization Strategy (Resolved from .env via Config)
    quant_config = None
    if cfg.QUANTIZATION == "4b":
        print("Applying 4-bit NormalFloat (NF4) quantization...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.USE_BF16 else torch.float16,
            bnb_4bit_quant_type="nf4"
        )
    elif cfg.QUANTIZATION == "8b":
        print("Applying 8-bit quantization...")
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16 if cfg.USE_BF16 else torch.float16,
        )
    else:
        print("No quantization applied. Loading model in full precision (bf16/fp16).")

    # 4. Tokenizer Initialization
    tokenizer = AutoTokenizer.from_pretrained(cfg.BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 5. Base Model Initialization
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.BASE_MODEL,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype= torch.bfloat16
    )
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    # 6. Prepare Model for LoRA Configuration
    if quant_config is not None:
        base_model = prepare_model_for_kbit_training(base_model)

    lora_parameters = LoraConfig(
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        r=cfg.LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.TARGET_MODULES,
    )

    # 7. Define Training Parameters (SFTConfig)
    train_parameters = SFTConfig(
        output_dir=cfg.PROJECT_RUN_NAME,
        num_train_epochs=cfg.EPOCHS,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
        optim=cfg.OPTIMIZER,
        save_steps=cfg.SAVE_STEPS,
        save_total_limit=10,
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
        hub_private_repo=True,
        eval_strategy="steps",
        eval_steps=cfg.SAVE_STEPS,
        gradient_checkpointing=True,
    )

    # 8. Initialize Trainer Engine
    fine_tuning = SFTTrainer(
        model=base_model,
        processing_class=tokenizer,
        train_dataset=train,
        eval_dataset=val,
        peft_config=lora_parameters,
        args=train_parameters,
        formatting_func=formatting_prompts_func
    )

    # 9. Execute Training
    print("Starting SFT training process...")
    fine_tuning.train()

    # 10. Save and Push Adapters
    if cfg.PUSH_TO_HUB:
        print(f"Pushing trained model to HF Hub: {cfg.PROJECT_RUN_NAME}...")
        fine_tuning.model.push_to_hub(cfg.PROJECT_RUN_NAME, private=True)

    if cfg.LOG_TO_WANDB:
        wandb.finish()