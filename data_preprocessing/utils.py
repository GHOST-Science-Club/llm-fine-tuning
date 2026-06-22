import asyncio
from pathlib import Path
from openai import AsyncOpenAI, APIError
from .config import config, PipelineConfig
from datasets import load_dataset, DatasetDict


class LLMClient:
    """
    Async wrapper around an OpenAI-compatible endpoint (e.g. vLLM).

    Owns a single AsyncOpenAI client (shared connection pool) and a semaphore
    that caps the number of concurrent in-flight requests. Use as an async
    context manager so the underlying HTTP client is always closed:
    """

    def __init__(self, config: PipelineConfig):
        self._cfg = config
        self._sem = asyncio.Semaphore(config.MAX_CONCURRENCY)


        self._client = AsyncOpenAI(
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
            timeout=config.REQUEST_TIMEOUT,
        )

    async def call(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM asynchronously, respecting the concurrency limit."""
        async with self._sem:
            try:
                response = await self._client.chat.completions.create(
                    model=self._cfg.MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self._cfg.TEMPERATURE,
                )
            except APIError as e:
                if self._cfg.DEBUG:
                    print(f"\n[API ERROR] model={self._cfg.MODEL}")
                    print(f"[API ERROR] {e}")
                raise

        return response.choices[0].message.content.strip()

    async def aclose(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

def debug(stage: str, content: str) -> None:
    """Print a labelled debug block when DEBUG=true."""
    if not config.DEBUG:
        return
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  [DEBUG] {stage}")
    print(sep)
    print(content)
    print(sep)


def save_dataset(
    input_file_path: str | Path,
    output_destination: str | Path,
    val_size: float = 0.1,
    seed: int = config.SEED,
) -> None:
    """Saves HuggingFace dataset locally (Path) or pushes to HF Hub (str repo id), with train/val split."""
    input_path = Path(input_file_path)
    if not input_path.exists() or input_path.stat().st_size == 0:
        print(f"Skipping dataset save: {input_path} is empty or does not exist.")
        return

    raw = load_dataset("json", data_files=str(input_path), split="train")
    splits = raw.train_test_split(test_size=val_size, seed=seed)
    dataset = DatasetDict({"train": splits["train"], "validation": splits["test"]})

    if isinstance(output_destination, Path):
        dataset.save_to_disk(output_destination)
        print(f"Dataset saved locally in: {output_destination} "
              f"(train={len(dataset['train'])}, validation={len(dataset['validation'])})")
    elif isinstance(output_destination, str):
        dataset.push_to_hub(output_destination)
        print(f"Dataset pushed to HF Hub: {output_destination} "
              f"(train={len(dataset['train'])}, validation={len(dataset['validation'])})")
    else:
        raise ValueError("output_destination must be a local Path or a HF Hub repo id string.")




