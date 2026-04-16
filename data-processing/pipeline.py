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
import sys
from latex_utils import normalize_latex
from utils import debug, save_dataset
from stages import split_tasks, filter_question, fix_latex_solution, find_answer, rewrite_answer, classify_question
from pathlib import Path
from config import INPUT_DIR, OUTPUT_FILE, DEBUG, DATASET_FILE

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
            "category": None,
            "category_reason": None,
            "raw_answer": None,
            "answer_post_index": None,
            "solution": None,
        }

        if not filt["keep"]:
            print(f"    -> DISCARDED: {filt['reason']}")
            results.append(record)
            continue

        print(f"    -> KEPT. Classifying question...")
        classification = classify_question(question)
        category = classification.get("category")
        category_reason = classification.get("reason")

        if category is None:
            print(f"    -> Invalid or missing category! Discarding.")
            results.append(record)
            continue

        record['category'], record['category_reason'] = category, category_reason
        print(f"Category found: {category}, reason: {category_reason}")

        print(f"    -> Finding answer (inline={has_inline})...")
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

    save_dataset(str(OUTPUT_FILE), str(DATASET_FILE))

if __name__ == "__main__":
    main()