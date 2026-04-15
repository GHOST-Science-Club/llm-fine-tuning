"""
Fine-tuning data pipeline.

Steps per thread:
  0. Split thread into individual tasks (a thread may contain multiple problems)
  1. Filter out unsuitable tasks (geometry requiring drawing, spam, off-topic)
  2. Find the correct answer within the forum posts
  3. Rewrite the answer step-by-step with full chain-of-thought reasoning

Output: pipeline_output.jsonl  (one record per task)

Usage:
  python pipeline.py                        # process all files in output/
  python pipeline.py path/to/file.jsonl     # process a single file
  DEBUG=1 python pipeline.py ...            # enable verbose stage output


keep in mind - we use pure latex (maybe better to change as it is norm in llms?)
"""
import glob
import json
import os
import re
import sys
from pathlib import Path
import requests
from dotenv import load_dotenv
import prompts
load_dotenv()

API_KEY = os.getenv("PCSS_API_KEY", "")
BASE_URL = os.getenv("PCSS_BASE_URL", "https://llm.hpc.psnc.pl/v1/chat/completions")
MODEL = os.getenv("MODEL", "llama3.3:70b")
DEBUG = os.getenv("DEBUG", "false") == "true"

INPUT_DIR = Path(__file__).parent / "data" / "input"
OUTPUT_FILE = Path(__file__).parent / "data" / "output" / "pipeline_output.jsonl"

def debug(stage: str, content: str) -> None:
    """Print a labelled debug block when DEBUG=true."""
    if not DEBUG:
        return
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  [DEBUG] {stage}")
    print(sep)
    print(content)
    print(sep)


# ---------------------------------------------------------------------------
# LaTeX normalisation
# ---------------------------------------------------------------------------

def _fix_displaystyle_blocks(text: str) -> str:
    """
    Replace \\(\\displaystyle{...}\\) with \\[...\\].
    Uses a brace counter so nested {} inside the expression are handled correctly.
    """
    marker = '\\(\\displaystyle{'   # literal: \(\displaystyle{
    result = []
    i = 0
    while i < len(text):
        pos = text.find(marker, i)
        if pos == -1:
            result.append(text[i:])
            break
        result.append(text[i:pos])
        j = pos + len(marker)
        depth = 1
        while j < len(text) and depth > 0:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        inner = text[pos + len(marker): j - 1].strip()
        if text[j: j + 2] == '\\)':
            result.append(f'\\[\n{inner}\n\\]')
            i = j + 2
        else:
            # No closing \) — still fix \displaystyle{} syntax
            result.append('{\\displaystyle ' + inner + '}')
            i = j
    return ''.join(result)


def normalize_latex(text: str) -> str:
    """
    Normalise LaTeX scraped from forums into clean, compilable LaTeX.

    Fixes applied:
      1. Unicode whitespace (non-breaking spaces etc.) → regular space
      2. \\(\\displaystyle{...}\\) → \\[...\\]
      3. Remaining \\(...\\) inline math → $...$
      4. Empty superscripts  ^{}  removed
      5. Leading spaces inside index braces  _{  x  → _{x
      6. Bare ...  →  \\cdots
    """
    # 1. Unicode whitespace variants
    text = text.replace('\u00a0', ' ')   # non-breaking space
    text = text.replace('\u2009', ' ')   # thin space
    text = text.replace('\u200b', '')    # zero-width space

    # 2. \(\displaystyle{...}\) → \[...\]
    text = _fix_displaystyle_blocks(text)

    # 3. Remaining \(...\) → $...$
    text = re.sub(r'\\\((.+?)\\\)', r'$\1$', text, flags=re.DOTALL)

    # 4. Remove empty superscripts
    text = text.replace('^{}', '')

    # 5. Clean leading spaces inside _{ and ^{
    text = re.sub(r'([_^])\{\s+', r'\1{', text)

    # 6. Bare ... → \cdots  (not preceded by a dot or backslash)
    text = re.sub(r'(?<![.\\])\.\.\.', r'\\cdots', text)

    return text

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Call the LLM API (OpenAI-compatible).
    Swap BASE_URL / auth headers here when moving to PCSS.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    response = requests.post(
        BASE_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

# ---------------------------------------------------------------------------
# Step 0 — Split tasks
# ---------------------------------------------------------------------------


def split_tasks(title: str, posts: list[dict]) -> list[dict]:
    """
    Returns list of {"question": str, "post_indices": list[int]}.
    Falls back to treating the whole thread as one task on parse errors.
    """
    posts_text = "\n".join(
        f"  [{p['index']}] author: {p['author']} — {p['content'][:500]}"
        for p in posts
    )
    user_prompt = f"Thread title: {title}\n\nPosts:\n{posts_text}"

    raw = call_llm(prompts.SPLIT_SYSTEM, user_prompt)
    debug("STEP 0 — split_tasks | raw LLM output", raw)

    try:
        # Strip markdown code fences if present
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        tasks = json.loads(clean)
        if isinstance(tasks, list) and tasks:
            debug("STEP 0 — split_tasks | parsed tasks", json.dumps(tasks, ensure_ascii=False, indent=2))
            return tasks
    except (json.JSONDecodeError, KeyError):
        debug("STEP 0 — split_tasks | JSON parse failed, using fallback", raw)

    # Fallback: one task = first post as question
    return [{"question": posts[0]["content"] if posts else title, "post_indices": list(range(len(posts)))}]


# ---------------------------------------------------------------------------
# Step 1 — Filter
# ---------------------------------------------------------------------------

def filter_question(question: str, relevant_posts: list[dict] | None = None) -> dict:
    """Returns {"keep": bool, "reason": str}."""
    has_images = any(p.get("contains_images", False) for p in (relevant_posts or []))
    user_prompt = (
        f"Problem: {question[:1000]}\n"
        f"contains_images in relevant posts: {str(has_images).lower()}"
    )
    raw = call_llm(prompts.FILTER_SYSTEM, user_prompt)
    debug("STEP 1 — filter_question | raw LLM output", raw)

    keep = True
    reason = ""
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("DECISION:"):
            keep = "YES" in line.upper()
        elif line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    debug("STEP 1 — filter_question | parsed result", f"keep={keep}  reason={reason}")
    return {"keep": keep, "reason": reason}

# ---------------------------------------------------------------------------
# Step 2 — Find correct answer
# ---------------------------------------------------------------------------

def find_answer(question: str, posts: list[dict], relevant_indices: list[int],
                has_inline_solution: bool = False) -> tuple[str | None, int | None]:
    """
    Returns (answer_text, source_post_index), or (None, None) if no answer found.
    has_inline_solution=True hints that the answer may be inside the question post itself.
    """
    relevant = [p for p in posts if p["index"] in relevant_indices]
    posts_text = "\n".join(
        f"  [{p['index']}] {p['author']} (contains_images={p.get('contains_images', False)}): "
        f"{p['content'][:800]}"
        for p in relevant
    )
    hint = (
        " Note: this problem post already contains the worked solution — extract it from there."
        if has_inline_solution else ""
    )
    user_prompt = (
        f"Problem: {question[:1000]}\n\n"
        f"Posts:\n{posts_text}\n\n"
        f"Find the best answer.{hint}"
    )

    raw = call_llm(prompts.FIND_ANSWER_SYSTEM, user_prompt).strip()
    debug("STEP 2 — find_answer | raw LLM output", raw)

    if "NO_ANSWER" in raw or not raw:
        debug("STEP 2 — find_answer | result", "NO ANSWER FOUND")
        return None, None

    # Parse POST_INDEX — LLM only returns the index, we fetch the full post ourselves
    post_index = None
    for line in raw.splitlines():
        if line.startswith("POST_INDEX:"):
            try:
                post_index = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass

    if post_index is None:
        debug("STEP 2 — find_answer | result", "Could not parse POST_INDEX")
        return None, None

    # Return the FULL original post content (not the LLM's copy of it)
    post_map = {p["index"]: p for p in posts}
    answer_post = post_map.get(post_index)
    if answer_post is None:
        debug("STEP 2 — find_answer | result", f"POST_INDEX {post_index} not found in posts")
        return None, None

    answer_text = answer_post["content"]
    debug("STEP 2 — find_answer | parsed result", f"post_index={post_index}\n{answer_text[:300]}")
    return answer_text, post_index


# ---------------------------------------------------------------------------
# Step 3 — Rewrite answer (chain-of-thought)
# ---------------------------------------------------------------------------

def rewrite_answer(question: str, raw_answer: str) -> str:
    user_prompt = (
        f"Problem: {question[:1000]}\n\n"
        f"Raw answer: {raw_answer[:2000]}\n\n"
        "Rewritten solution:"
    )
    result = call_llm(prompts.REWRITE_SYSTEM, user_prompt).strip()
    debug("STEP 3 — rewrite_answer | full rewritten solution", result)
    return result

# ---------------------------------------------------------------------------
# Step 4 — Fix LaTeX (final LLM pass)
# ---------------------------------------------------------------------------

def fix_latex_solution(solution: str) -> str:
    """Final LLM pass to catch any remaining LaTeX syntax errors in the solution."""
    result = call_llm(prompts.FIX_LATEX_SYSTEM, solution).strip()
    # Strip markdown code fences in case the model wraps its output
    result = result.removeprefix('```latex').removeprefix('```').removesuffix('```').strip()
    debug("STEP 4 — fix_latex_solution | corrected output", result)
    return result

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_thread(thread: dict) -> list[dict]:
    """Process one thread, returning a list of result records (one per task)."""
    title = thread.get("title", "")
    url = thread.get("url", "")
    posts = thread.get("posts", [])

    # Normalise LaTeX in all post content before any LLM call
    posts = [
        {**p, "content": normalize_latex(p.get("content", ""))}
        for p in posts
    ]
    debug("PRE — normalize_latex | first post after normalisation",
          posts[0]["content"][:300] if posts else "(no posts)")

    print(f"  Splitting tasks...")
    tasks = split_tasks(title, posts)
    print(f"  Found {len(tasks)} task(s)")

    results = []
    for i, task in enumerate(tasks):
        question = normalize_latex(task["question"])
        relevant_indices = task.get("post_indices", list(range(len(posts))))
        has_inline = task.get("has_inline_solution", False)
        relevant_posts = [p for p in posts if p["index"] in relevant_indices]

        print(f"  Task {i+1}/{len(tasks)}: filtering...")
        filt = filter_question(question, relevant_posts)

        print(f"    -> Cleaning question LaTeX...")
        question_clean = fix_latex_solution(question)

        record = {
            "source_url": url,
            "source_title": title,
            "question": question_clean,
            "kept": filt["keep"],
            "filter_reason": filt["reason"],
            "raw_answer": None,
            "answer_post_index": None,
            "solution": None,
        }

        if not filt["keep"]:
            print(f"    -> DISCARDED: {filt['reason']}")
            results.append(record)
            continue

        print(f"    -> KEPT. Finding answer (inline={has_inline})...")
        raw_answer, answer_post_idx = find_answer(question, posts, relevant_indices, has_inline)

        if raw_answer is None:
            print(f"    -> No answer found in thread")
            results.append(record)
            continue

        print(f"    -> Cleaning answer LaTeX...")
        record["raw_answer"] = fix_latex_solution(normalize_latex(raw_answer))
        record["answer_post_index"] = answer_post_idx

        print(f"    -> Rewriting answer...")
        solution = rewrite_answer(question_clean, raw_answer)

        print(f"    -> Fixing solution LaTeX...")
        record["solution"] = fix_latex_solution(solution)

        results.append(record)

    return results


def main():
    # Optional single-file argument: python pipeline.py path/to/file.jsonl
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if not target.exists():
            print(f"File not found: {target}")
            sys.exit(1)
        jsonl_files = [str(target)]
        print(f"Single-file mode: {target}")
    else:
        jsonl_files = glob.glob(str(INPUT_DIR / "*.jsonl"))
        if not jsonl_files:
            print(f"No JSONL files found in {INPUT_DIR}")
            return
        print(f"Found {len(jsonl_files)} file(s) in {INPUT_DIR}")

    if DEBUG:
        print("[DEBUG mode ON]")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for filepath in jsonl_files:
            print(f"\nProcessing: {Path(filepath).name}")
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    thread = json.loads(line)
                    records = process_thread(thread)
                    for record in records:
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()