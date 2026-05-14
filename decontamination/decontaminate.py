"""
Dataset decontamination using datatrove's built-in NGramsDecontFilter.

Two-stage pipeline:
  Stage 1 — build_index():
      Reads benchmark .jsonl files and writes a binary hash index to disk.
      Uses the same datatrove utility functions (simplify_text, word_tokenize,
      ngrams, hash_func) as datatrove's own NGramsDecontIndexer — just without
      the lighteval dependency (we supply our own benchmark data directly).

  Stage 2 — run_filter():
      Runs a datatrove pipeline:
        LocalHFDatasetReader → NGramsDecontFilter → JsonlWriter
      The filter loads the index built in stage 1.

Usage:
  python decontaminate.py
  python decontaminate.py --datasets ../data/dataset/processed --benchmarks ../data/benchmark.jsonl
  python decontaminate.py --ngram-size 9 --threshold 1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.decont.n_grams import NGramsDecontConfig, NGramsDecontFilter
from datatrove.pipeline.writers import JsonlWriter
from datatrove.utils.hashing import HashConfig, create_hash_func
from datatrove.utils.text import TextNormConfig, ngrams, simplify_text
from datatrove.utils.word_tokenizers import load_word_tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from readers import make_dataset_readers

ROOT = Path(__file__).parent.parent
DEFAULT_DATASET_PATH = str(ROOT / "data" / "dataset" / "processed")
DEFAULT_BENCHMARK_PATH = str(ROOT / "data" / "benchmark.jsonl")
DEFAULT_OUTPUT_DIR = str(ROOT / "data" / "decontaminated")
DEFAULT_INDEX_DIR = str(ROOT / "data" / "decont_index")


# ── Stage 1: build index ────────────────────────────────────────────────────────

def _is_hf_dataset_dir(path: Path) -> bool:
    return (path / "dataset_info.json").exists()


def _safe_json_loads(line: str) -> dict:
    """
    Parse a JSON line that may contain bare LaTeX backslashes (e.g. \{, \dot, \neq)
    which are invalid JSON escape sequences. Falls back to escaping them if needed.
    """
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        # Valid JSON escapes: " \ / b f n r t uXXXX
        # Anything else after \ is invalid — escape the backslash
        valid = set('"\\\/bfnrtu')
        fixed = []
        i = 0
        while i < len(line):
            if line[i] == '\\' and i + 1 < len(line) and line[i + 1] not in valid:
                fixed.append('\\\\')
            else:
                fixed.append(line[i])
            i += 1
        return json.loads(''.join(fixed))


def _iter_benchmark_rows(
    benchmark_path: str,
    task_key: str,
    solution_key: str,
) -> list[dict]:
    """
    Yields dicts from a benchmark regardless of format:
      - .jsonl file
      - local HF dataset (save_to_disk format)
      - HF Hub repo_id (e.g. "org/dataset-name")
    """
    from datasets import load_from_disk, load_dataset, DatasetDict

    p = Path(benchmark_path)

    if p.exists() and p.suffix in (".jsonl", ".json"):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield _safe_json_loads(line)

    elif p.exists() and _is_hf_dataset_dir(p):
        ds = load_from_disk(str(p))
        if isinstance(ds, DatasetDict):
            ds = ds["train"]
        yield from ds

    else:
        # assume HF Hub repo_id
        ds = load_dataset(benchmark_path, split="train")
        yield from ds


def build_index(
    benchmark_paths: list[str],
    index_dir: str,
    benchmark_task_key: str = "task",
    benchmark_solution_key: str = "solution",
    ngram_size: int = 9,
) -> None:
    """
    Reads benchmarks (JSONL, local HF dataset, or HF Hub) and writes one
    .index.hashes binary file per benchmark task so NGramsDecontFilter can
    report exactly which benchmark task a removed training example matched.
    """
    config = NGramsDecontConfig(n_grams=ngram_size, find_query_ngrams=True)
    tokenizer = load_word_tokenizer("en")
    hash_func = create_hash_func(config.hash_config)
    norm_config = config.norm_config

    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    for benchmark_path in benchmark_paths:
        benchmark_name = Path(benchmark_path).stem
        count = 0

        for item in _iter_benchmark_rows(benchmark_path, benchmark_task_key, benchmark_solution_key):
            solution = item.get(benchmark_solution_key, "") or ""
            task_text = item.get(benchmark_task_key, "") or ""
            task_id = item.get("task_id", None)

            # one index file per task → contaminated_task will show exact task_id
            task_name = f"{benchmark_name}_task{task_id}" if task_id is not None else f"{benchmark_name}_{count}"

            label_tokens = tokenizer.word_tokenize(simplify_text(solution, norm_config))
            query_tokens = tokenizer.word_tokenize(simplify_text(task_text, norm_config))

            ngrams_to_hash = list(ngrams(label_tokens, ngram_size))
            ngrams_to_hash += list(ngrams(query_tokens, ngram_size))
            ngrams_to_hash += [
                query_tokens[-ngram_size + 1 + i:] + label_tokens[:i + 1]
                for i in range(ngram_size - 1)
                if len(query_tokens) >= ngram_size - 1 - i and len(label_tokens) >= i + 1
            ]

            hash_set: set[int] = {hash_func(" ".join(gram)) for gram in ngrams_to_hash}

            out_file = index_path / f"{task_name}.index.hashes"
            hashes_array = np.array(list(hash_set), dtype=np.dtype(config.hash_config.np_descr))
            with open(out_file, "wb") as f:
                hashes_array.tofile(f)

            count += 1

        print(f"Index built for '{benchmark_name}': {count} tasks")


# ── Stage 2: filter ─────────────────────────────────────────────────────────────

def run_filter(
    dataset_paths: list[str],
    index_dir: str,
    output_dir: str,
    ngram_size: int,
    limit: int,
) -> None:
    """
    Runs datatrove pipeline: LocalHFDatasetReader → NGramsDecontFilter → JsonlWriter.
    Training doc.text = 'question' field (checked against benchmark n-gram index).
    """
    output_path = Path(output_dir)
    jsonl_output = str(output_path / "clean")
    removed_output = str(output_path / "removed")

    config = NGramsDecontConfig(n_grams=ngram_size, find_query_ngrams=True)
    dataset_readers = make_dataset_readers(dataset_paths, text_key=["question", "raw_answer"], limit=limit)

    for i, reader in enumerate(dataset_readers):
        source = reader.path if hasattr(reader, "path") else reader.repo_id
        print(f"\n=== Filtering dataset {i+1}/{len(dataset_readers)}: {source} ===")

        executor = LocalPipelineExecutor(
            pipeline=[
                reader,
                NGramsDecontFilter(
                    index_folder=index_dir,
                    config=config,
                    exclusion_writer=JsonlWriter(removed_output),
                ),
                JsonlWriter(jsonl_output),
            ],
            tasks=1,
            logging_dir=str(output_path / "logs"),
        )
        executor.run()

    # Convert clean JSONL output → HF dataset with flat column structure
    jsonl_files = list(Path(jsonl_output).glob("*.jsonl*"))
    if jsonl_files:
        from datasets import load_dataset as hf_load
        raw_ds = hf_load("json", data_files=[str(f) for f in jsonl_files], split="train")

        # collect all metadata keys across rows so every row has the same columns
        all_keys = set()
        for row in raw_ds:
            all_keys.update((row.get("metadata") or {}).keys())
        all_keys.discard("_source_path")

        def flatten_row(row):
            flat = dict(row.get("metadata") or {})
            flat.pop("_source_path", None)
            for k in all_keys:
                flat.setdefault(k, None)
            return flat

        clean_ds = raw_ds.map(flatten_row, remove_columns=raw_ds.column_names)
        hf_output = str(output_path / "hf_dataset")
        clean_ds.save_to_disk(hf_output)
        print(f"\nClean HF dataset saved to: {hf_output}  ({len(clean_ds)} examples)")
        print(f"Columns: {clean_ds.column_names}")
    else:
        print("\nNo clean documents produced.")

    print(f"Removed documents: {removed_output}")


# ── entry point ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=[DEFAULT_DATASET_PATH])
    parser.add_argument("--benchmarks", nargs="+", default=[DEFAULT_BENCHMARK_PATH])
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ngram-size", type=int, default=9,
                        help="N-gram size (default: 9, same as datatrove's default)")
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    print("=== Stage 1: Building benchmark index ===")
    build_index(
        benchmark_paths=args.benchmarks,
        index_dir=args.index_dir,
        ngram_size=args.ngram_size,
    )

    print("\n=== Stage 2: Filtering training data ===")
    run_filter(
        dataset_paths=args.datasets,
        index_dir=args.index_dir,
        output_dir=args.output_dir,
        ngram_size=args.ngram_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
