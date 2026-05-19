import sys
import os
import json

eval_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, eval_dir)
os.chdir(eval_dir)

import lm_eval
import torch
from lm_eval.utils import make_table
from lm_eval.tasks import TaskManager

if __name__ == "__main__":
    device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on: {device_to_use}")

    task_manager = TaskManager(include_path=eval_dir)

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args="pretrained=speakleash/Bielik-11B-v3.0-Instruct",
        tasks=["benchmark"],
        task_manager=task_manager,
        apply_chat_template=True,
        device=device_to_use,
        batch_size="auto",
        log_samples=True,
        fewshot_as_multiturn=True
    )

    with open("logs/predictions.json", "w", encoding="utf-8") as f:
        json.dump(results["samples"], f, ensure_ascii=False, indent=2)

    print(make_table(results))