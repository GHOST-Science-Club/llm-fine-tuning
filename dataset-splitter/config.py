import argparse
from dataclasses import dataclass
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = MODULE_ROOT / "data"
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_FORMATTED_DIR = DEFAULT_DATA_DIR / "formatted"


@dataclass
class Config:
    """Configuration settings for dataset splitting."""

    input_file: Path | str = DEFAULT_FORMATTED_DIR / "dataset.jsonl"
    sft_file: Path | str = DEFAULT_FORMATTED_DIR / "sft.jsonl"
    grpo_file: Path | str = DEFAULT_FORMATTED_DIR / "grpo.jsonl"
    sft_ratio: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        """Ensures all file paths are converted to absolute Path objects."""
        self.input_file = Path(self.input_file).resolve()
        self.sft_file = Path(self.sft_file).resolve()
        self.grpo_file = Path(self.grpo_file).resolve()

    def ensure_directories(self) -> None:
        """Ensures that the output directories exist on disk."""
        self.input_file.parent.mkdir(parents=True, exist_ok=True)
        self.sft_file.parent.mkdir(parents=True, exist_ok=True)
        self.grpo_file.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> Config:
    """Parses command-line arguments and returns a Config instance with absolute paths."""
    default_config = Config()

    parser = argparse.ArgumentParser(
        description="Split a JSONL dataset into SFT and GRPO subsets randomly."
    )
    parser.add_argument(
        "--input",
        "-i",
        default=str(default_config.input_file),
        help=f"Path to input file (default: {default_config.input_file})",
    )
    parser.add_argument(
        "--sft_out",
        default=str(default_config.sft_file),
        help=f"Path to SFT output file (default: {default_config.sft_file})",
    )
    parser.add_argument(
        "--grpo_out",
        default=str(default_config.grpo_file),
        help=f"Path to GRPO output file (default: {default_config.grpo_file})",
    )
    parser.add_argument(
        "--sft_ratio",
        type=float,
        default=default_config.sft_ratio,
        help=f"Proportion of the SFT dataset (default: {default_config.sft_ratio})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_config.seed,
        help=f"Random seed for reproducibility (default: {default_config.seed})",
    )

    args = parser.parse_args()

    return Config(
        input_file=args.input,
        sft_file=args.sft_out,
        grpo_file=args.grpo_out,
        sft_ratio=args.sft_ratio,
        seed=args.seed,
    )
