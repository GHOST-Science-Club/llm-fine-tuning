# Data Preprocessing

Turns raw math-forum threads  into a clean, structured dataset which can be used in SFT/GRPO training. Each thread is split into individual problems, filtered, classified, matched with the best answer, rewritten into clean LaTeX, and graded for correctness through a sequence of LLM calls.

The pipeline produces two results:

* **Clean dataset** — only training-ready fields (`source_url`, `question`, `category`, `solution`, `final_answer`), split into `train`/`val` and saved as a HuggingFace `datasets` dataset (locally or pushed to the Hub).
* **Diagnostic log** — a separate JSONL with every record and all intermediate metadata (filter decisions, reasons, raw answers, grading reasons), for inspection. Optional.

---

## Setup

### 1. Install dependencies

From the repository root:

```bash
pip install -r data_preprocessing/requirements.txt
```

### 2. Create `.env`

Configuration is read from a `.env` file at the module directory.

| Variable          | `.env.example` value         | Description                                                             |
|-------------------|-------------------------------|-------------------------------------------------------------------------|
| `API_KEY`         | `your_api_key_here`          | Bearer token for the LLM API — replace with your own                    |
| `API_BASE_URL`    | `https://llm.hpc.psnc.pl/v1` | OpenAI-compatible base URL (the SDK appends `/chat/completions`)         |
| `MODEL`           | `ai_model`                   | Model name sent in the request payload — replace with your own          |
| `DEBUG`           | `false`                       | `true` prints verbose per-step LLM input/output                         |
| `LOAD_FROM_HUB`   | `false`                       | `true` loads input from a HF Hub dataset instead of the local JSONL     |
| `PUSH_TO_HUB`     | `false`                       | `true` pushes the clean dataset to the HF Hub instead of saving locally |
| `SAVE_LOGS`       | `false`                       | `false` disables the diagnostic log file entirely                       |
| `MAX_CONCURRENCY` | `32`                           | Maximum number of concurrent in-flight requests to the LLM API          |
| `BATCH_SIZE`      | `128`                          | Threads processed per concurrent batch; checkpoint is written per batch |

> Omitting a variable (or the whole `.env` file) falls back to `config.py`'s built-in defaults, which differ from the template above for a few fields: `API_KEY` → empty, `MODEL` → `llama3.3:70b`, `SAVE_LOGS` → `true`, `MAX_CONCURRENCY` → `8`, and `BATCH_SIZE` → `0`, which auto-computes to `2 × MAX_CONCURRENCY`.

> If you push to the Hub (`PUSH_TO_HUB=true`) or load a gated dataset (`LOAD_FROM_HUB=true`), make sure you are logged in (`huggingface-cli login`) or have `HF_TOKEN` set in your environment.

---

## Running

Run as a module **from the repository root**:

```bash
python -m data_preprocessing.pipeline
```

Suppress the per-step console output:

```bash
python -m data_preprocessing.pipeline --quiet
```

On completion the pipeline prints statistics (loaded / kept / filtered out / parse errors) and the paths of the saved artifacts.

---

## How it works

Each thread flows through the following steps, one LLM call per step per task:

| Step | Method | What it does |
|---|---|---|
| 0 | `_split_tasks` | Splits a thread into separate problems and the post indices relevant to each |
| — | `normalize_latex` | Rule-based LaTeX cleanup (no LLM), applied before any prompt |
| 1 | `_filter_question` | Keeps solvable text/LaTeX problems; discards drawing-/image-dependent or off-topic ones |
| 2 | `_fix_latex` | LLM pass converting implicit-math notation into clean `$…$` / `\[…\]` — reused for the question, the raw answer, and the rewritten solution |
| 3 | `_classify_question` | Assigns one category: `EXACT_VALUE`, `EXPRESSION`, `PROOF`, `COMPLEX` |
| 4 | `_find_correct_answer` | Picks the post index with the most complete, correct answer |
| — | `_token_overlap` | Rule-based check (no LLM): discards an answer that's just the question restated |
| 5 | `_rewrite_answer` | Restructures the answer into numbered steps (`Krok 1:`, …) ending with `\textbf{Wynik:}` |
| 6 | `_grade_solution` | Final quality gate — rejects incorrect/incomplete solutions and extracts the standalone `final_answer` |

A task only reaches the clean dataset if it survives all checks (kept → valid category → answer found → not a restated question → rewritten → graded valid). Every task — including discarded ones — is written to the diagnostic log with the reason.

### Concurrency

The pipeline runs on `asyncio`. Threads are processed in **concurrent batches** of `BATCH_SIZE`, and within each thread its tasks run concurrently too. The steps *inside* a single task stay sequential (each depends on the previous one). All LLM calls go through a shared async client whose semaphore caps the number of in-flight requests at `MAX_CONCURRENCY`, regardless of how many tasks are queued.

Because many threads run at once, per-step console logs interleave — use `--quiet` for clean output, and keep `DEBUG=false` on real runs.

### Checkpointing

Progress is saved to `data/checkpoint/checkpoint.txt` as `next_thread_idx:0`. The checkpoint advances only after a whole batch finishes, so resuming restarts at a batch boundary and never re-emits already-written records. Restarting resumes from the checkpoint and appends to the existing output files. Delete the checkpoint file to start fresh.

---

## Input / output format

**Input** — one thread per JSONL line:

```json
{
  "url": "https://matematyka.pl/viewtopic.php?t=412551",
  "title": "Ciekawy iloczyn",
  "posts": [
    {"index": 0, "author": "user", "content": "\\text{Udowodnić, że }...", "contains_images": false}
  ]
}
```

**Clean record** (in the dataset):

```json
{"source_url": "...", "question": "...", "category": "PROOF", "solution": "Krok 1: ...", "final_answer": "..."}
```

**Full record** (in the log) additionally includes `source_title`, `kept`, `filter_reason`, `category_reason`, `raw_answer`, `answer_post_index`, and `grading_reason`.

The clean dataset is split **90% train / 10% val** before saving (see `save_dataset` in `utils.py`).

---

## Project structure

```text
data_preprocessing/
├── pipeline.py        # CLI entry point (argparse, builds the pipeline from config)
├── models.py          # DataProcessingPipeline — orchestration + record building
├── prompts.py         # LLM system prompts (few-shot, structured outputs)
├── latex_utils.py     # Rule-based normalize_latex() preprocessing
├── utils.py           # LLMClient (async vLLM/OpenAI-compatible), debug(), save_dataset()
├── config.py          # PipelineConfig dataclass — paths, env vars, flags
├── requirements.txt   # Python dependencies
└── data/
    ├── input/         # Source JSONL threads
    ├── output/        # Clean records (feeds the dataset)
    ├── logs/          # Full diagnostic records (optional)
    ├── dataset/       # Saved HuggingFace dataset (train/val)
    └── checkpoint/    # Resume checkpoint
```
