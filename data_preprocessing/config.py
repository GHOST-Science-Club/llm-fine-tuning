import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv


_MODULE_PATH = Path(__file__).resolve().parent
load_dotenv(_MODULE_PATH / ".env", override=True)


@dataclass
class PipelineConfig:
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("API_BASE_URL", "https://llm.hpc.psnc.pl/v1"))
    model: str = field(default_factory=lambda: os.getenv("MODEL", "llama3.3:70b"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false") == "true")
    load_from_hub: bool = field(default_factory=lambda: os.getenv("LOAD_FROM_HUB", "false") == "true")
    push_to_hub: bool = field(default_factory=lambda: os.getenv("PUSH_TO_HUB", "false") == "true")
    save_logs: bool = field(default_factory=lambda: os.getenv("SAVE_LOGS", "true") == "true")

    data_dir: Path = field(default_factory=lambda: _MODULE_PATH / "data")

    # Local input/output paths, relative to data_dir unless an absolute path is given.
    input_file_name: str = field(default_factory=lambda: os.getenv("INPUT_FILE", "input/forum_example_fixed.jsonl"))
    output_file_name: str = field(default_factory=lambda: os.getenv("OUTPUT_FILE", "output/pipeline_output.jsonl"))

    input_file: Path = field(init=False)
    output_file: Path = field(init=False)
    log_file: Path | None = field(init=False)
    dataset_file: Path = field(init=False)
    checkpoint_file: Path = field(init=False)

    # Hugging Face Hub dataset ids used when load_from_hub / push_to_hub are enabled.
    hub_dataset_source: str = field(default_factory=lambda: os.getenv("HUB_DATASET_SOURCE", "meta-math/MetaMathQA"))
    hub_dataset_destination: str = field(default_factory=lambda: os.getenv("HUB_DATASET_DESTINATION", "erybie222/test"))

    # input_source and dataset_destination can be either a local file or a Hugging Face Hub dataset, depending on load_from_hub and push_to_hub flags
    input_source: str | Path = field(init=False)
    dataset_destination: str | Path = field(init=False)

    # Hyperparameters for LLM calls and dataset processing
    temperature: float = 0.2
    request_timeout: int = 60
    seed: int = 42
    # Max number of concurrent in-flight LLM requests (asyncio semaphore bound).
    max_concurrency: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENCY", "8")))
    # How many threads are processed per concurrent batch. Checkpoint is written
    # after each batch completes. 0 → defaults to 2 * max_concurrency in __post_init__.
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "0")))


    # What fields make up the clean, training-ready dataset (everything else is diagnostic).
    clean_fields : tuple[str, ...]= field(init=False)
    # Word-overlap (Jaccard) ratio above which a "found answer" is treated as just the
    # question restated rather than a real solution, and discarded.
    answer_overlap_threshold : float = field(init=False)
    question_length_threshold : int = field(init=False)
    solution_length_threshold : int = field(init=False)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            self.batch_size = 2 * self.max_concurrency
        self.input_file = self.data_dir / self.input_file_name
        self.output_file = self.data_dir / self.output_file_name
        self.log_file = self.data_dir / "logs" / "pipeline_logs.jsonl" if self.save_logs else None
        self.dataset_file = self.data_dir / "dataset" / "pipeline_output.jsonl"
        self.checkpoint_file = self.data_dir / "checkpoint" / "checkpoint.txt"
        self.input_source = self.hub_dataset_source if self.load_from_hub else self.input_file
        self.dataset_destination = self.hub_dataset_destination if self.push_to_hub else self.dataset_file
        self.clean_fields = ("source_url", "question", "category", "solution", "final_answer")
        self.answer_overlap_threshold = 0.85
        self.question_length_threshold = 1000
        self.solution_length_threshold = 3000
        paths = [self.output_file.parent, self.dataset_file.parent,
                 self.input_file.parent, self.checkpoint_file.parent]
        if self.log_file:
            paths.append(self.log_file.parent)
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)


config = PipelineConfig()
