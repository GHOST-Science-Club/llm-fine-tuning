import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv


_MODULE_PATH = Path(__file__).resolve().parent
load_dotenv(_MODULE_PATH / ".env", override=True)


@dataclass
class PipelineConfig:
    API_KEY: str = field(default_factory=lambda: os.getenv("PCSS_API_KEY", ""))
    BASE_URL: str = field(default_factory=lambda: os.getenv("PCSS_BASE_URL", "https://llm.hpc.psnc.pl/v1/chat/completions"))
    MODEL: str = field(default_factory=lambda: os.getenv("MODEL", "llama3.3:70b"))
    DEBUG: bool = field(default_factory=lambda: os.getenv("DEBUG", "false") == "true")
    LOAD_FROM_HUB: bool = field(default_factory=lambda: os.getenv("LOAD_FROM_HUB", "false") == "true")
    PUSH_TO_HUB: bool = field(default_factory=lambda: os.getenv("PUSH_TO_HUB", "false") == "true")
    SAVE_LOGS: bool = field(default_factory=lambda: os.getenv("SAVE_LOGS", "true") == "true")

    DATA_DIR: Path = field(default_factory=lambda: _MODULE_PATH / "data")

    INPUT_FILE: Path = field(init=False)
    OUTPUT_FILE: Path = field(init=False)
    LOG_FILE: Path | None = field(init=False)
    CHECKPOINT_FILE: Path = field(init=False)

    # INPUT_SOURCE and DATASET_DESTINATION can be either a local file or a Hugging Face Hub dataset, depending on LOAD_FROM_HUB and PUSH_TO_HUB flags
    INPUT_SOURCE: str | Path = field(init=False)
    DATASET_DESTINATION: str | Path = field(init=False)

    # Hyperparameters for LLM calls and dataset processing
    TEMPERATURE: float = 0.2
    REQUEST_TIMEOUT: int = 60
    SEED: int = 42

    def __post_init__(self) -> None:
        self.INPUT_FILE = self.DATA_DIR / "input" / "forum_example_fixed.jsonl"
        self.OUTPUT_FILE = self.DATA_DIR / "output" / "pipeline_output.jsonl"
        self.LOG_FILE = self.DATA_DIR / "logs" / "pipeline_logs.jsonl" if self.SAVE_LOGS else None
        self.DATASET_FILE = self.DATA_DIR / "dataset" / "pipeline_output.jsonl"
        self.CHECKPOINT_FILE = self.DATA_DIR / "checkpoint" / "checkpoint.txt"
        self.INPUT_SOURCE = "meta-math/MetaMathQA" if self.LOAD_FROM_HUB else self.INPUT_FILE
        self.DATASET_DESTINATION = "erybie222/test" if self.PUSH_TO_HUB else self.DATASET_FILE
        paths = [self.OUTPUT_FILE.parent, self.DATASET_FILE.parent,
                 self.INPUT_FILE.parent, self.CHECKPOINT_FILE.parent]
        if self.LOG_FILE:
            paths.append(self.LOG_FILE.parent)
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)


config = PipelineConfig()
