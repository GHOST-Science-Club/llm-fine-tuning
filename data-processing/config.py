import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

API_KEY = os.getenv("PCSS_API_KEY", "")
BASE_URL = os.getenv("PCSS_BASE_URL", "https://llm.hpc.psnc.pl/v1/chat/completions")
MODEL = os.getenv("MODEL", "llama3.3:70b")
DEBUG = os.getenv("DEBUG", "false") == "true"

INPUT_DIR = Path(__file__).parent / "data" / "input"
OUTPUT_FILE = Path(__file__).parent / "data" / "output" / "pipeline_output.jsonl"
DATASET_FILE = Path(__file__).parent / "data" / "dataset" / "pipeline_output.jsonl"