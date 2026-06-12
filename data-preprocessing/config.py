import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv


def _resolve_data_dir() -> Path:
    module_path = Path(__file__).resolve().parent
    load_dotenv(module_path.parents[0] / ".env", override=True)
    return module_path / "data"


@dataclass
class PipelineConfig:
    API_KEY: str = field(default_factory=lambda: os.getenv("PCSS_API_KEY", ""))
    BASE_URL: str = field(default_factory=lambda: os.getenv("PCSS_BASE_URL", "https://llm.hpc.psnc.pl/v1/chat/completions"))
    MODEL: str = field(default_factory=lambda: os.getenv("MODEL", "llama3.3:70b"))
    DEBUG: bool = field(default_factory=lambda: os.getenv("DEBUG", "false") == "true")
    LOAD_FROM_HUB: bool = field(default_factory=lambda: os.getenv("LOAD_FROM_HUB", "false") == "true")

    DATA_DIR: Path = field(default_factory=_resolve_data_dir)

    INPUT_FILE: Path = field(init=False)
    OUTPUT_FILE: Path = field(init=False)
    DATASET_FILE: Path = field(init=False)
    CHECKPOINT_FILE: Path = field(init=False)
    INPUT_SOURCE: str | Path = field(init=False)

    def __post_init__(self) -> None:
        self.INPUT_FILE = self.DATA_DIR / "input" / "forum_example_fixed.jsonl"
        self.OUTPUT_FILE = self.DATA_DIR / "output" / "pipeline_output.jsonl"
        self.DATASET_FILE = self.DATA_DIR / "dataset" / "pipeline_output.jsonl"
        self.CHECKPOINT_FILE = self.DATA_DIR / "checkpoint" / "checkpoint.txt"
        self.INPUT_SOURCE = "meta-math/MetaMathQA" if self.LOAD_FROM_HUB else self.INPUT_FILE
        for path in [self.OUTPUT_FILE.parent, self.DATASET_FILE.parent,
                     self.INPUT_FILE.parent, self.CHECKPOINT_FILE.parent]:
            path.mkdir(parents=True, exist_ok=True)


config = PipelineConfig()
