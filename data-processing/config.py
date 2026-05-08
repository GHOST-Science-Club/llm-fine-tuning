import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_PATH = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_PATH / ".env", override=True)

API_KEY = os.getenv("PCSS_API_KEY", "")
BASE_URL = os.getenv("PCSS_BASE_URL", "https://llm.hpc.psnc.pl/v1/chat/completions")
MODEL = os.getenv("MODEL", "llama3.3:70b")
DEBUG = os.getenv("DEBUG", "false") == "true"

DATA_DIR = ROOT_PATH / "data-processing" / "data"

load_dotenv(ROOT_PATH / ".env", override=True)

INPUT_FILE = DATA_DIR / "input" / "forum_example_fixed.jsonl"
OUTPUT_FILE = DATA_DIR / "output" / "pipeline_output.jsonl"
DATASET_FILE = DATA_DIR / "dataset" / "pipeline_output.jsonl"
CHECKPOINT_FILE = DATA_DIR / "checkpoint" / "checkpoint.txt"

for path in [OUTPUT_FILE.parent, DATASET_FILE.parent, INPUT_FILE.parent]:
    path.mkdir(parents=True, exist_ok=True)