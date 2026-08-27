from pathlib import Path
import random
from .config import Config



def split_dataset(config: Config) -> None:
    """Splits a dataset into SFT and GRPO subsets based on the provided configuration."""
    input_path = Path(config.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    print(f"Loading data from: {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    total_count = len(lines)
    print(f"Total lines in dataset: {total_count}")

    # Set random seed for reproducibility
    random.seed(config.seed)
    random.shuffle(lines)

    sft_count = int(total_count * config.sft_ratio)
    grpo_count = total_count - sft_count

    sft_lines = lines[:sft_count]
    grpo_lines = lines[sft_count:]

    sft_path = Path(config.sft_file)
    grpo_path = Path(config.grpo_file)

    print(
        f"Saving SFT ({len(sft_lines)} lines, {config.sft_ratio * 100:.1f}%) to: {sft_path}..."
    )
    with open(sft_path, "w", encoding="utf-8") as f:
        f.writelines(sft_lines)

    print(
        f"Saving GRPO ({len(grpo_lines)} lines, {(1 - config.sft_ratio) * 100:.1f}%) to: {grpo_path}..."
    )
    with open(grpo_path, "w", encoding="utf-8") as f:
        f.writelines(grpo_lines)

    print("\nDataset splitting completed successfully!")
    print(f" - SFT:   {len(sft_lines):>6} samples ({sft_path.name})")
    print(f" - GRPO:  {len(grpo_lines):>6} samples ({grpo_path.name})")
    print(f" - Total: {total_count:>6} samples")
