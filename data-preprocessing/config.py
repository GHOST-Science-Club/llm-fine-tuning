import os
from pathlib import Path
from dotenv import load_dotenv

MODULE_PATH = Path(__file__).resolve().parents[0]
load_dotenv(MODULE_PATH / ".env", override=True)

API_KEY = os.getenv("PCSS_API_KEY", "")
BASE_URL = os.getenv("PCSS_BASE_URL", "https://llm.hpc.psnc.pl/v1/chat/completions")
MODEL = os.getenv("MODEL", "llama3.3:70b")
DEBUG = os.getenv("DEBUG", "false") == "true"
LOAD_FROM_HUB = os.getenv("LOAD_FROM_HUB", "false") == "true"

DATA_DIR = MODULE_PATH / "data-preprocessing" / "data"
INPUT_FILE = DATA_DIR / "input" / "forum_example_fixed.jsonl"
OUTPUT_FILE = DATA_DIR / "output" / "pipeline_output.jsonl"
DATASET_FILE = DATA_DIR / "dataset" / "pipeline_output.jsonl"
CHECKPOINT_FILE = DATA_DIR / "checkpoint" / "checkpoint.txt"

if LOAD_FROM_HUB:
    INPUT_SOURCE = "meta-math/MetaMathQA"
else:
    INPUT_SOURCE = Path(INPUT_FILE)

for path in [OUTPUT_FILE.parent, DATASET_FILE.parent, INPUT_FILE.parent, CHECKPOINT_FILE.parent]:
    path.mkdir(parents=True, exist_ok=True)