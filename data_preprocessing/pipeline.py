from .models import DataProcessingPipeline
from .config import config
from .utils import LLMClient
import argparse
import asyncio


async def _run(quiet: bool) -> None:
    # The LLMClient owns the async HTTP client; `async with` guarantees it is
    # closed when the pipeline finishes (or raises).
    async with LLMClient(config) as llm:
        pipeline = DataProcessingPipeline(config, llm=llm, quiet=quiet)
        await pipeline.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--quiet', action='store_true', help="Hiding console print statements while running pipeline")
    args = parser.parse_args()

    asyncio.run(_run(args.quiet))

if __name__ == "__main__":
    main()
