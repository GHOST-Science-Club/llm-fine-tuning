from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv


_MODULE_PATH = Path(__file__).resolve().parent
_REPO_PATH = _MODULE_PATH.parent
load_dotenv(_MODULE_PATH / ".env", override=True)


@dataclass
class VerificationConfig:
    API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    # None lets the OpenAI SDK use its own default endpoint (api.openai.com).
    BASE_URL: str | None = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL") or None)
    MODEL: str = field(default_factory=lambda: os.getenv("MODEL", "gpt-5-mini"))
    DEBUG: bool = field(default_factory=lambda: os.getenv("DEBUG", "false") == "true")

    # How many records to sample from the input file, and the seed used to pick them.
    # Same seed + same input file -> same examples, so a re-run is comparable.
    NUM_EXAMPLES: int = field(default_factory=lambda: int(os.getenv("NUM_EXAMPLES", "30")))
    SEED: int = field(default_factory=lambda: int(os.getenv("SEED", "42")))

    # Max number of concurrent in-flight API requests (asyncio semaphore bound).
    MAX_CONCURRENCY: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENCY", "5")))
    # Reasoning models are slower to first token than plain chat models.
    REQUEST_TIMEOUT: int = 120

    # Both the input dataset and the verification results live under the repo-root data/ dir.
    data_dir: Path = field(default_factory=lambda: _REPO_PATH / "data")

    INPUT_FILE: Path = field(init=False)
    OUTPUT_FILE: Path = field(init=False)

    def __post_init__(self) -> None:
        self.INPUT_FILE = self.data_dir / "2026_08_14_data_preprocessing_output.jsonl"
        self.OUTPUT_FILE = self.data_dir / "verification" / "verification_results.jsonl"
        self.OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


config = VerificationConfig()
