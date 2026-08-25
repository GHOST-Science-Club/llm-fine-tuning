from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI

from .config import config, VerificationConfig
from .prompts import VERIFY_SYSTEM


# The four dimensions the model is asked to check. Anything else it returns is dropped.
KNOWN_ISSUES = ("math", "question", "final_answer", "latex")
VERDICTS = ("CORRECT", "MINOR_ISSUES", "INCORRECT")
# Pseudo-verdicts for records we could not judge; reported separately in the summary.
FAILURES = ("PARSE_ERROR", "API_ERROR")

SEP_HEAVY = "═" * 78
SEP_LIGHT = "─" * 78


class Judge:
    """
    Async wrapper around the OpenAI API for one-shot verification calls.

    Owns a single AsyncOpenAI client (shared connection pool) and a semaphore
    that caps the number of concurrent in-flight requests. Use as an async
    context manager so the underlying HTTP client is always closed.
    """

    def __init__(self, config: VerificationConfig):
        self._cfg = config
        self._sem = asyncio.Semaphore(config.MAX_CONCURRENCY)
        self._client = AsyncOpenAI(
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
            timeout=config.REQUEST_TIMEOUT,
            max_retries=3,
        )

    async def call(self, user_prompt: str) -> str:
        """Call the model asynchronously, respecting the concurrency limit."""
        async with self._sem:
            # No temperature / max_tokens: reasoning models (gpt-5*) reject both,
            # and the defaults are what we want for every model anyway.
            response = await self._client.chat.completions.create(
                model=self._cfg.MODEL,
                messages=[
                    {"role": "system", "content": VERIFY_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
            )

        if not response.choices:
            raise ValueError(f"Model returned no choices (model={self._cfg.MODEL})")

        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"Model returned empty content (model={self._cfg.MODEL})")

        return content.strip()

    async def aclose(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> "Judge":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()


def load_records(input_file: str | Path) -> list[dict]:
    """Read the pipeline's JSONL output into a list of records."""
    with open(input_file, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_records(records: list[dict], num_examples: int, seed: int) -> list[dict]:
    """Pick `num_examples` records at random; the same seed always picks the same ones."""
    return random.Random(seed).sample(records, min(num_examples, len(records)))


def build_user_prompt(record: dict) -> str:
    """Assemble the record into the block the system prompt expects."""
    return (
        f"CATEGORY: {record.get('category', 'UNKNOWN')}\n"
        f"QUESTION:\n{record.get('question', '')}\n"
        f"SOLUTION:\n{record.get('solution', '')}\n"
        f"FINAL_ANSWER: {record.get('final_answer') or 'NONE'}"
    )


def parse_reply(raw: str) -> dict:
    """Parse the VERDICT / ISSUES / COMMENT lines. Unknown verdict -> PARSE_ERROR."""
    verdict = "PARSE_ERROR"
    issues: list[str] = []
    comment_lines: list[str] = []
    in_comment = False

    for line in raw.splitlines():
        line = line.strip().strip("*").strip()
        upper = line.upper()
        if upper.startswith("VERDICT:"):
            in_comment = False
            if "INCORRECT" in upper:
                verdict = "INCORRECT"
            elif "MINOR" in upper:
                verdict = "MINOR_ISSUES"
            elif "CORRECT" in upper:
                verdict = "CORRECT"
        elif upper.startswith("ISSUES:"):
            in_comment = False
            tags = line.split(":", 1)[1].lower().replace(" ", "").split(",")
            issues = [tag for tag in tags if tag in KNOWN_ISSUES]
        elif upper.startswith("COMMENT:"):
            in_comment = True
            comment_lines.append(line.split(":", 1)[1].strip())
        elif in_comment and line:
            comment_lines.append(line)

    comment = " ".join(comment_lines).strip()
    return {
        "verdict": verdict,
        "issues": issues,
        "comment": None if comment.upper() in ("", "NONE") else comment,
    }


def _label(index: int, total: int) -> str:
    """`[ 7/30]` — the tag that identifies one example across every line we print."""
    return f"[{index:>{len(str(total))}}/{total}]"


def format_result(index: int, total: int, record: dict, result: dict, raw: str) -> str:
    """
    Render everything we print about one example as a single string.

    Examples are judged concurrently, so each one must be printed in a single
    print() call — otherwise the lines of different examples interleave and it
    becomes impossible to tell which output belongs to which problem.
    """
    label = _label(index, total)
    header = f"{label} {result['verdict']:<13} {record.get('category', 'UNKNOWN')}"
    issues = ", ".join(result["issues"])

    if not config.DEBUG:
        lines = [f"{header:<40} {record.get('source_url', '?')}"]
        if result["comment"]:
            prefix = f"{issues}: " if issues else ""
            lines.append(f"{' ' * len(label)} └─ {prefix}{result['comment']}")
        return "\n".join(lines)

    return "\n".join([
        "",
        SEP_HEAVY,
        f" {header}" + (f"   [{issues}]" if issues else ""),
        SEP_HEAVY,
        " QUESTION",
        record.get("question", ""),
        "",
        " SOLUTION",
        record.get("solution", ""),
        "",
        f" FINAL_ANSWER: {record.get('final_answer') or 'NONE'}",
        "",
        " MODEL REPLY",
        raw or f"(no reply — {result['comment']})",
        SEP_LIGHT,
    ])


async def verify_one(judge: Judge, index: int, total: int, record: dict) -> dict:
    """Verify a single record. API/parse failures become a verdict, never an exception."""
    raw = ""
    try:
        raw = await judge.call(build_user_prompt(record))
        result = parse_reply(raw)
    except Exception as e:
        result = {"verdict": "API_ERROR", "issues": [], "comment": f"{type(e).__name__}: {e}"}

    print(format_result(index, total, record, result, raw))
    return {"sample_index": index, **record, **result}


def print_summary(results: list[dict], input_file: Path, output_file: Path) -> None:
    """Print verdict counts, issue frequencies, a per-category table and what to review."""
    total = len(results)
    verdicts = Counter(r["verdict"] for r in results)
    issues = Counter(issue for r in results for issue in r["issues"])
    categories = sorted({r.get("category", "UNKNOWN") for r in results})
    # Only show the failure columns/rows when something actually failed.
    shown = VERDICTS + tuple(v for v in FAILURES if verdicts[v])

    print(f"\n{SEP_HEAVY}\n VERIFICATION SUMMARY\n{SEP_HEAVY}")
    print(f" Model    {config.MODEL}")
    print(f" Seed     {config.SEED}")
    print(f" Checked  {total} example(s)")
    print(f" Input    {input_file}")
    print(f" Output   {output_file}")

    print("\n VERDICTS")
    for verdict in shown:
        count = verdicts[verdict]
        share = count / total if total else 0
        print(f"   {verdict:<13} {count:>4}  {share:>6.1%}  {'█' * round(share * 30)}")

    print("\n ISSUES")
    for issue, count in issues.most_common() or [("none", 0)]:
        print(f"   {issue:<13} {count:>4}")

    print("\n BY CATEGORY")
    print(f"   {'category':<13} {'total':>5}" + "".join(f" {v:>13}" for v in shown))
    for category in categories:
        counts = Counter(r["verdict"] for r in results if r.get("category", "UNKNOWN") == category)
        row = "".join(f" {counts[v]:>13}" for v in shown)
        print(f"   {category:<13} {sum(counts.values()):>5}{row}")

    flagged = [r for r in results if r["verdict"] != "CORRECT"]
    if flagged:
        print("\n EXAMPLES TO REVIEW")
        for r in sorted(flagged, key=lambda r: (VERDICTS + FAILURES).index(r["verdict"])):
            label = _label(r["sample_index"], total)
            print(f"\n   {label} {r['verdict']:<13} {r.get('category', 'UNKNOWN')}")
            print(f"   {' ' * len(label)} {r.get('source_url', '?')}")
            if r["issues"]:
                print(f"   {' ' * len(label)} issues: {', '.join(r['issues'])}")
            if r["comment"]:
                print(f"   {' ' * len(label)} {r['comment']}")
    print(f"\n{SEP_LIGHT}")


async def _run(input_file: Path, output_file: Path) -> None:
    records = load_records(input_file)
    sample = sample_records(records, config.NUM_EXAMPLES, config.SEED)
    total = len(sample)
    print(f"Verifying {total} of {len(records)} examples with {config.MODEL} "
          f"(seed {config.SEED}, up to {config.MAX_CONCURRENCY} at a time)")
    print(f"Input: {input_file}\n")

    # The Judge owns the async HTTP client; `async with` guarantees it is closed.
    async with Judge(config) as judge:
        results = await asyncio.gather(
            *(verify_one(judge, i, total, record) for i, record in enumerate(sample, start=1))
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print_summary(results, input_file, output_file)


def main():
    parser = argparse.ArgumentParser(description="Verify the correctness of preprocessed math examples with the OpenAI API.")
    parser.add_argument('-n', '--num-examples', type=int, default=config.NUM_EXAMPLES, help="How many randomly sampled examples to verify")
    parser.add_argument('-m', '--model', default=config.MODEL, help="OpenAI model to use as the judge")
    parser.add_argument('-s', '--seed', type=int, default=config.SEED, help="Sampling seed; the same seed picks the same examples")
    parser.add_argument('-i', '--input', type=Path, default=config.INPUT_FILE, help="Input JSONL produced by the preprocessing pipeline")
    parser.add_argument('-o', '--output', type=Path, default=config.OUTPUT_FILE, help="Where to write the per-example results")
    args = parser.parse_args()

    if not config.API_KEY:
        raise SystemExit("OPENAI_API_KEY is not set. Copy verification/.env.example to verification/.env and fill it in.")

    config.NUM_EXAMPLES = args.num_examples
    config.MODEL = args.model
    config.SEED = args.seed

    asyncio.run(_run(args.input, args.output))


if __name__ == "__main__":
    main()
