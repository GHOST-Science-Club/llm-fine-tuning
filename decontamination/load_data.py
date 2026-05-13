"""
Load training datasets and benchmarks into datatrove Documents.

Usage examples:
  # defaults (local dataset + local benchmark)
  python load_data.py

  # custom paths
  python load_data.py \
    --datasets ../data/dataset/processed \
    --benchmarks ../data/benchmark.jsonl

  # multiple datasets and benchmarks
  python load_data.py \
    --datasets ../data/dataset/processed GHOST-Science-Club/my-other-dataset \
    --benchmarks ../data/benchmark.jsonl ../data/benchmark2.jsonl

  # override column names (if your data uses different field names)
  python load_data.py --dataset-text-key question --benchmark-text-key task
"""

import argparse
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.writers import JsonlWriter

from readers import make_dataset_readers, make_benchmark_readers

ROOT = Path(__file__).parent.parent
DEFAULT_DATASET_PATH = str(ROOT / "data" / "dataset" / "processed")
DEFAULT_BENCHMARK_PATH = str(ROOT / "data" / "benchmark.jsonl")
DEFAULT_OUTPUT_DIR = str(ROOT / "data" / "decontamination_output")


def load_and_preview( # for testing of corectness of data loading
    dataset_paths: list[str],
    benchmark_paths: list[str],
    dataset_text_key: str,
    benchmark_text_key: str,
    benchmark_id_key: str,
    output_dir: str,
    limit: int,
) -> None:
    dataset_readers = make_dataset_readers(dataset_paths, text_key=dataset_text_key, limit=limit)
    benchmark_readers = make_benchmark_readers(benchmark_paths, text_key=benchmark_text_key, id_key=benchmark_id_key, limit=limit)

    print(f"\n=== Loaded {len(dataset_readers)} training dataset(s), {len(benchmark_readers)} benchmark(s) ===\n")

    print("--- Training datasets ---")
    for reader in dataset_readers:
        docs = list(reader.run())
        print(f"  {reader.path if hasattr(reader, 'path') else reader.repo_id}: {len(docs)} documents")
        if docs:
            d = docs[0]
            print(f"    text preview : {d.text[:120]!r}")
            print(f"    metadata keys: {list(d.metadata.keys())}")

    print("\n--- Benchmarks ---")
    for reader in benchmark_readers:
        executor = LocalPipelineExecutor(pipeline=[reader], tasks=1)
        # preview only — collect a few docs manually
        docs = []
        for doc in reader.run():
            docs.append(doc)
            if len(docs) >= 3:
                break
        print(f"  {reader.data_folder}: {reader} — first {len(docs)} docs sampled")
        if docs:
            d = docs[0]
            print(f"    text preview : {d.text[:120]!r}")
            print(f"    metadata keys: {list(d.metadata.keys())}")

    print("\n=== Data loading complete. Ready for decontamination. ===")


def main():
    parser = argparse.ArgumentParser(description="Load training datasets and benchmarks.")
    parser.add_argument(
        "--datasets", nargs="+",
        default=[DEFAULT_DATASET_PATH],
        help="Local HF dataset path(s) or HF Hub repo_id(s). Folders are searched recursively for HF datasets.",
    )
    parser.add_argument(
        "--benchmarks", nargs="+",
        default=[DEFAULT_BENCHMARK_PATH],
        help="Path(s) to benchmark .jsonl file(s) or folders containing .jsonl files.",
    )
    parser.add_argument("--dataset-text-key", default="question", help="Column name for the main text in training datasets.")
    parser.add_argument("--benchmark-text-key", default="task", help="Column name for the main text in benchmarks.")
    parser.add_argument("--benchmark-id-key", default="task_id", help="Column name for the id in benchmarks.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Where to write output (used in later decontamination step).")
    parser.add_argument("--limit", type=int, default=-1, help="Max documents to load per source (useful for testing).")
    args = parser.parse_args()

    load_and_preview(
        dataset_paths=args.datasets,
        benchmark_paths=args.benchmarks,
        dataset_text_key=args.dataset_text_key,
        benchmark_text_key=args.benchmark_text_key,
        benchmark_id_key=args.benchmark_id_key,
        output_dir=args.output_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
