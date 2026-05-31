"""
Dataset deduplication using datatrove's MinHash LSH pipeline.

Four-stage pipeline:
  Stage 1 — MinhashDedupSignature:
      Reads training dataset and writes MinHash signatures to disk.

  Stage 2 — MinhashDedupBuckets:
      Groups signatures into LSH buckets to find candidate duplicate pairs.

  Stage 3 — MinhashDedupCluster:
      Clusters duplicate pairs using union-find; each cluster gets one representative.

  Stage 4 — MinhashDedupFilter:
      Reads training dataset again; keeps one document per cluster, removes the rest.
      Removed documents are written to a separate JSONL for inspection.

Usage:
  python deduplicate.py
  python deduplicate.py --datasets ../data/dataset/processed
  python deduplicate.py --ngram-size 5 --num-buckets 14 --hashes-per-bucket 8
  python deduplicate.py --limit 100  # quick test run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "utils_decontamination_deduplication"))

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupBuckets, MinhashDedupCluster, MinhashDedupFilter, MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import MinhashConfig
from datatrove.pipeline.writers import JsonlWriter

from readers import make_dataset_readers

ROOT = Path(__file__).parent.parent
DEFAULT_DATASET_PATH = str(ROOT / "data" / "dataset" / "processed")
DEFAULT_OUTPUT_DIR = str(ROOT / "data" / "deduplicated")


def run_dedup(
    dataset_paths: list[str],
    output_dir: str,
    text_key: list[str],
    ngram_size: int,
    num_buckets: int,
    hashes_per_bucket: int,
    tasks: int,
    limit: int,
) -> None:
    """
    Runs the full 4-stage MinHash deduplication pipeline.
    Input:  one or more HF datasets (local or Hub)
    Output: clean HF dataset + removed JSONL, both with flat original column names.
    """
    output_path = Path(output_dir)
    sig_dir       = str(output_path / "sigs")
    bucket_dir    = str(output_path / "buckets")
    cluster_dir   = str(output_path / "clusters")
    jsonl_clean   = str(output_path / "clean")
    jsonl_removed = str(output_path / "removed")
    logs_sig      = str(output_path / "logs" / "stage1_sig")
    logs_buckets  = str(output_path / "logs" / "stage2_buckets")
    logs_cluster  = str(output_path / "logs" / "stage3_cluster")
    logs_filter   = str(output_path / "logs" / "stage4_filter")

    if tasks % num_buckets != 0:
        raise ValueError(f"--tasks ({tasks}) must be divisible by --num-buckets ({num_buckets})")

    config = MinhashConfig(
        n_grams=ngram_size,
        num_buckets=num_buckets,
        hashes_per_bucket=hashes_per_bucket,
    )

    dataset_readers_sig = make_dataset_readers(dataset_paths, text_key=text_key, limit=limit)
    dataset_readers_filter = make_dataset_readers(dataset_paths, text_key=text_key, limit=limit)

    # spawn works on both Windows and Linux; forkserver (datatrove default) fails on Windows
    start_method = "spawn" if tasks > 1 else None

    # ── Stage 1: compute MinHash signatures ──────────────────────────────────
    print("=== Stage 1: Computing MinHash signatures ===")
    for i, reader in enumerate(dataset_readers_sig):
        source = getattr(reader, "path", None) or getattr(reader, "data_folder", None) or getattr(reader, "repo_id", None)
        print(f"  [{i+1}/{len(dataset_readers_sig)}] {source}")
        LocalPipelineExecutor(
            pipeline=[
                reader,
                MinhashDedupSignature(output_folder=sig_dir, config=config),
            ],
            tasks=tasks,
            start_method=start_method,
            logging_dir=logs_sig,
        ).run()

    # ── Stage 2: bucket candidate pairs ─────────────────────────────────────
    print("\n=== Stage 2: Grouping into LSH buckets ===")
    LocalPipelineExecutor(
        pipeline=[MinhashDedupBuckets(input_folder=sig_dir, output_folder=bucket_dir, config=config)],
        tasks=tasks,
        start_method=start_method,
        logging_dir=logs_buckets,
    ).run()

    # ── Stage 3: cluster duplicate pairs ────────────────────────────────────
    print("\n=== Stage 3: Clustering duplicate pairs ===")
    LocalPipelineExecutor(
        pipeline=[MinhashDedupCluster(input_folder=bucket_dir, output_folder=cluster_dir, config=config)],
        tasks=1,
        logging_dir=logs_cluster,
    ).run()

    # ── Stage 4: filter — keep one doc per cluster ───────────────────────────
    print("\n=== Stage 4: Filtering duplicates ===")
    cluster_files = list(Path(cluster_dir).glob("*.remove")) if Path(cluster_dir).exists() else []
    no_duplicates = len(cluster_files) == 0
    if no_duplicates:
        print("  No duplicate clusters found — forwarding all documents unchanged.")

    for i, reader in enumerate(dataset_readers_filter):
        source = getattr(reader, "path", None) or getattr(reader, "data_folder", None) or getattr(reader, "repo_id", None)
        print(f"  [{i+1}/{len(dataset_readers_filter)}] {source}")
        if no_duplicates:
            # skip dedup filter — just copy everything to clean output
            LocalPipelineExecutor(
                pipeline=[reader, JsonlWriter(jsonl_clean)],
                tasks=1,
                logging_dir=logs_filter,
            ).run()
        else:
            LocalPipelineExecutor(
                pipeline=[
                    reader,
                    MinhashDedupFilter(
                        input_folder=cluster_dir,
                        exclusion_writer=JsonlWriter(jsonl_removed),
                    ),
                    JsonlWriter(jsonl_clean),
                ],
                tasks=1,
                logging_dir=logs_filter,
            ).run()

    # ── Convert clean JSONL → HF dataset with flat column structure ──────────
    jsonl_files = list(Path(jsonl_clean).glob("*.jsonl*"))
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

    print(f"Removed documents (duplicates): {jsonl_removed}")


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate training datasets using MinHash LSH."
    )
    parser.add_argument("--datasets", nargs="+", default=[DEFAULT_DATASET_PATH],
                        help="Local HF dataset path(s) or HF Hub repo_id(s).")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory for all outputs (default: data/deduplicated).")
    parser.add_argument("--text-key", nargs="+", default=["question", "raw_answer"],
                        help="Dataset column(s) concatenated for similarity comparison.")
    parser.add_argument("--ngram-size", type=int, default=3,
                        help="Shingle (n-gram) size for MinHash — smaller catches more paraphrases (default: 3).")
    parser.add_argument("--num-buckets", type=int, default=14,
                        help="Number of LSH buckets (default: 14). More buckets = better recall.")
    parser.add_argument("--tasks", type=int, default=14,
                        help="Number of parallel workers (default: 14). Must be divisible by --num-buckets.")
    parser.add_argument("--hashes-per-bucket", type=int, default=3,
                        help="Hashes per LSH bucket — lower = catches more near-duplicates (default: 3).")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Max documents per source, -1 for all (useful for testing).")
    args = parser.parse_args()

    run_dedup(
        dataset_paths=args.datasets,
        output_dir=args.output_dir,
        text_key=args.text_key,
        ngram_size=args.ngram_size,
        num_buckets=args.num_buckets,
        hashes_per_bucket=args.hashes_per_bucket,
        tasks=args.tasks,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
