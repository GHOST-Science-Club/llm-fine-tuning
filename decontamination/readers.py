"""
Custom datatrove readers for local HF datasets and flexible benchmark loading.

DataTrove Document fields:
  text     - main text content used for decontamination / deduplication comparisons
  id       - unique string id
  metadata - dict for all other columns
"""

from pathlib import Path
from typing import Optional

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import JsonlReader
from datasets import load_from_disk, load_dataset, DatasetDict


def _is_hf_dataset_dir(path: Path) -> bool:
    return (path / "dataset_info.json").exists()


class LocalHFDatasetReader(PipelineStep):
    """
    Reads an HF dataset saved with Dataset.save_to_disk() and yields datatrove Documents.
    Handles sharding automatically so it works with LocalPipelineExecutor(workers>1).
    """

    name = "LocalHFDatasetReader"
    _requires_dependencies = ["datasets"]

    def __init__(
        self,
        path: str,
        text_key: str | list[str] = "text",
        id_key: Optional[str] = None,
        split: str = "train",
        limit: int = -1,
    ):
        super().__init__()
        self.path = path
        self.text_keys = [text_key] if isinstance(text_key, str) else text_key
        self.id_key = id_key
        self.split = split
        self.limit = limit

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        ds = load_from_disk(self.path)
        if isinstance(ds, DatasetDict):
            ds = ds[self.split]

        if world_size > 1:
            ds = ds.shard(num_shards=world_size, index=rank, contiguous=True)

        metadata_keys = [c for c in ds.column_names if c not in (*self.text_keys, self.id_key)]

        for i, sample in enumerate(ds):
            if self.limit != -1 and i >= self.limit:
                break

            # concatenate all text fields so the filter checks all of them
            text = " ".join(sample.get(k) or "" for k in self.text_keys).strip()
            doc_id = (
                str(sample[self.id_key])
                if self.id_key and self.id_key in sample
                else f"{Path(self.path).name}_{rank}_{i}"
            )
            metadata = {k: sample[k] for k in metadata_keys if k in sample}
            # keep original text fields in metadata so the output can restore the flat structure
            for k in self.text_keys:
                if k in sample:
                    metadata[k] = sample[k]
            metadata["_source_path"] = self.path

            yield Document(text=text, id=doc_id, metadata=metadata)


class HubHFDatasetReader(PipelineStep):
    """
    Reads an HF dataset directly from the HuggingFace Hub via load_dataset().
    Uses streaming to avoid downloading the full dataset upfront.
    """

    name = "HubHFDatasetReader"
    _requires_dependencies = ["datasets"]

    def __init__(
        self,
        repo_id: str,
        text_key: str = "text",
        id_key: Optional[str] = None,
        split: str = "train",
        limit: int = -1,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.text_key = text_key
        self.id_key = id_key
        self.split = split
        self.limit = limit

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        ds = load_dataset(self.repo_id, split=self.split, streaming=True)
        # streaming datasets don't support shard(), so we skip every world_size-th sample
        for i, sample in enumerate(ds):
            if i % world_size != rank:
                continue
            if self.limit != -1 and i // world_size >= self.limit:
                break

            text = sample.get(self.text_key) or ""
            doc_id = (
                str(sample[self.id_key])
                if self.id_key and self.id_key in sample
                else f"{self.repo_id}_{i}"
            )
            metadata = {k: v for k, v in sample.items() if k not in (self.text_key, self.id_key)}
            metadata["_source_repo"] = self.repo_id

            yield Document(text=text, id=doc_id, metadata=metadata)


def make_benchmark_reader(
    path: str,
    text_key: str = "task",
    id_key: str = "task_id",
    limit: int = -1,
) -> JsonlReader:
    """
    Returns a datatrove JsonlReader for a benchmark .jsonl file or folder of .jsonl files.
    Uses an adapter to map arbitrary column names to the Document format.
    """

    def adapter(_reader, sample: dict, path: str, id_in_file: int | str) -> dict:
        return {
            "text": sample.get(text_key, ""),
            "id": str(sample.get(id_key, id_in_file)),
            "metadata": {k: v for k, v in sample.items() if k not in (text_key, id_key)},
        }

    return JsonlReader(data_folder=path, adapter=adapter, limit=limit)


def make_dataset_readers(
    paths: list[str],
    text_key: str | list[str] = "question",
    id_key: Optional[str] = None,
    limit: int = -1,
) -> list[LocalHFDatasetReader | HubHFDatasetReader]:
    """
    Returns a list of readers for one or more training datasets.
    Each path can be:
      - a local path to an HF dataset dir (has dataset_info.json)
      - a local path to a folder containing multiple HF dataset dirs
      - a HF Hub repo_id (e.g. "GHOST-Science-Club/my-dataset")
    """
    readers = []
    for path in paths:
        p = Path(path)

        if p.exists():
            if _is_hf_dataset_dir(p):
                readers.append(LocalHFDatasetReader(path, text_key=text_key, id_key=id_key, limit=limit))
            else:
                # folder containing multiple HF dataset subdirs
                subdirs = [d for d in p.iterdir() if d.is_dir() and _is_hf_dataset_dir(d)]
                if not subdirs:
                    raise ValueError(f"{path} is not an HF dataset dir and contains no HF dataset subdirs")
                for sub in sorted(subdirs):
                    readers.append(LocalHFDatasetReader(str(sub), text_key=text_key, id_key=id_key, limit=limit))
        else:
            # assume it's a HF Hub repo_id
            readers.append(HubHFDatasetReader(path, text_key=text_key, id_key=id_key, limit=limit))

    return readers


def make_benchmark_readers(
    paths: list[str],
    text_key: str = "task",
    id_key: str = "task_id",
    limit: int = -1,
) -> list[JsonlReader]:
    """
    Returns a list of JsonlReaders for one or more benchmark files or folders.
    Each path can be a .jsonl file or a folder containing .jsonl files.
    """
    readers = []
    for path in paths:
        p = Path(path)
        if p.is_file():
            readers.append(make_benchmark_reader(str(p.parent), text_key=text_key, id_key=id_key, limit=limit))
        elif p.is_dir():
            readers.append(make_benchmark_reader(str(p), text_key=text_key, id_key=id_key, limit=limit))
        else:
            raise FileNotFoundError(f"Benchmark path not found: {path}")
    return readers
