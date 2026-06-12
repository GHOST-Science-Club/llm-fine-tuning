from typing import Optional
from pathlib import Path
import requests
from .config import config
from datasets import load_dataset, DatasetDict

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Call the LLM API (OpenAI-compatible).
    Swap BASE_URL / auth headers here when moving to PCSS.
    """

    headers = {
        "Authorization": f"Bearer {config.API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    response = requests.post(
        config.BASE_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    if config.DEBUG and response.status_code != 200:
        print(f"\n[API ERROR] Payload: {payload}")
        print(f"[API ERROR] Server response: {response.text}")
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

def debug(stage: str, content: str) -> None:
    """Print a labelled debug block when DEBUG=true."""
    if not config.DEBUG:
        return
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  [DEBUG] {stage}")
    print(sep)
    print(content)
    print(sep)


def save_dataset(
    input_file_path: str | Path,
    output_destination: str | Path,
    val_size: float = 0.1,
    seed: int = 42,
) -> None:
    """Saves HuggingFace dataset locally (Path) or pushes to HF Hub (str repo id), with train/val split."""
    input_path = Path(input_file_path)
    if not input_path.exists() or input_path.stat().st_size == 0:
        print(f"Skipping dataset save: {input_path} is empty or does not exist.")
        return

    raw = load_dataset("json", data_files=str(input_path), split="train")
    splits = raw.train_test_split(test_size=val_size, seed=seed)
    dataset = DatasetDict({"train": splits["train"], "val": splits["test"]})

    if isinstance(output_destination, Path):
        dataset.save_to_disk(output_destination)
        print(f"Dataset saved locally in: {output_destination} "
              f"(train={len(dataset['train'])}, val={len(dataset['val'])})")
    elif isinstance(output_destination, str):
        dataset.push_to_hub(output_destination)
        print(f"Dataset pushed to HF Hub: {output_destination} "
              f"(train={len(dataset['train'])}, val={len(dataset['val'])})")
    else:
        raise ValueError("output_destination must be a local Path or a HF Hub repo id string.")




