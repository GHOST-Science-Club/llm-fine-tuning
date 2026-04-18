import json
from enum import Enum
from pathlib import Path
from utils import debug, call_llm, save_dataset
from prompts import SPLIT_SYSTEM, FILTER_SYSTEM, FIND_ANSWER_SYSTEM, CLASSIFY_SYSTEM, REWRITE_SYSTEM, FIX_LATEX_SYSTEM
from latex_utils import normalize_latex

class Category(str, Enum):
    EXACT_VALUE = "EXACT_VALUE"
    EXPRESSION = "EXPRESSION"
    PROOF = "PROOF"
    COMPLEX = "COMPLEX"

class DataProcessingPipeline:

    def __init__(self, input_file: Path, output_dir: Path, dataset_dir: Path):
        self.input_file = input_file
        self.output_dir = output_dir
        self.dataset_dir = dataset_dir
        self.raw_data = []
        self.processed_data = []
        self.stats = {
            "loaded": 0,
            "filtered_out": 0,
            "kept": 0,
            "llm_parse_errors": 0,
            "classification_errors": 0
        }

    def _load_data(self) -> None:
        """Loading raw data."""
        debug("Loading data", f"Starting to load data from file: {self.input_file}")

        with open(self.input_file, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    thread = json.loads(line)
                    self.raw_data.append(thread)
                    self.stats["loaded"] += 1

                except json.JSONDecodeError as e:
                    print(f"Warning: JSON parsing error at line {line_number}. Skipping this record. Details: {e}")

        debug("Loading data", f"Successfully loaded {self.stats['loaded']} records.")

    def _split_tasks(self, title: str, posts: list[dict]) -> list[dict]:
        """
        Helper method: Splits a thread into individual tasks/questions using an LLM.
        Returns a list of dicts: [{"question": str, "post_indices": list[int]}].
        Falls back to treating the whole thread as one task on parse errors.
        """
        posts_text = "\n".join(
            f"  [{p['index']}] author: {p['author']} — {p['content'][:500]}"
            for p in posts
        )
        user_prompt = f"Thread title: {title}\n\nPosts:\n{posts_text}"


        raw = call_llm(SPLIT_SYSTEM, user_prompt)
        debug("STEP 0 — split_tasks | raw LLM output", raw)

        try:
            clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            tasks = json.loads(clean)

            if isinstance(tasks, list) and tasks:
                debug("STEP 0 — split_tasks | parsed tasks", json.dumps(tasks, ensure_ascii=False, indent=2))
                return tasks

        except (json.JSONDecodeError, KeyError):
            debug("STEP 0 — split_tasks | JSON parse failed, using fallback", raw)
            self.stats["llm_parse_errors"] = self.stats.get("llm_parse_errors", 0) + 1

        return [{
            "question": posts[0]["content"] if posts else title,
            "post_indices": list(range(len(posts)))
        }]

    def _filter_question(self, question: str, relevant_posts: list[dict] | None = None) -> dict:
        """Returns {"keep": bool, "reason": str}."""
        has_images = any(p.get("contains_images", False) for p in (relevant_posts or []))
        user_prompt = (
            f"Problem: {question[:1000]}\n"
            f"contains_images in relevant posts: {str(has_images).lower()}"
        )
        raw = call_llm(FILTER_SYSTEM, user_prompt)
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

    def _classify_question(self, question: str) -> dict:
        """
        Assign category to the question based on the solution
        (exact value, expression, proof, complex)
        Returns {"category": Category | None, "reason": str}.
        """
        user_prompt = f"Problem: {question[:1000]}"
        raw = call_llm(CLASSIFY_SYSTEM, user_prompt)

        category_str = ""
        reason = ""

        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("CATEGORY:"):
                category_str = line.split(":", 1)[1].strip().upper()
            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        try:
            category = Category(category_str)
        except ValueError:
            category = None
            self.stats["classification_errors"] = self.stats.get("classification_errors", 0) + 1
            debug("STEP 2", f"WARNING: Invalid category from LLM: {category_str}")

        return {"category": category, "reason": reason}

    def _find_correct_answer(self, question: str, posts: list[dict], relevant_indices: list[int],
                    has_inline_solution: bool = False, ) -> tuple[str | None, int | None]:
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

        raw = call_llm(FIND_ANSWER_SYSTEM, user_prompt).strip()
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

    def _rewrite_answer(self, question: str, raw_answer: str) -> str:
        user_prompt = (
            f"Problem: {question[:1000]}\n\n"
            f"Raw answer: {raw_answer[:2000]}\n\n"
            "Rewritten solution:"
        )
        result = call_llm(REWRITE_SYSTEM, user_prompt).strip()
        debug("STEP 3 — rewrite_answer | full rewritten solution", result)
        return result

    def _fix_latex_solution(self, solution: str) -> str:
        """Final LLM pass to catch any remaining LaTeX syntax errors in the solution."""
        result = call_llm(FIX_LATEX_SYSTEM, solution).strip()
        # Strip markdown code fences in case the model wraps its output
        result = result.removeprefix('```latex').removeprefix('```').removesuffix('```').strip()
        debug("STEP 4 — fix_latex_solution | corrected output", result)
        return result

    def run(self) -> None:
        """
        Main pipeline method.
        Processes loaded data, applies LLM transformations/filters, and saves the output.
        """
        # 1. Load the raw data into self.raw_data
        self._load_data()

        # Define the output file path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / "pipeline_output.jsonl"

        print(f"\nStarting to process {len(self.raw_data)} threads...")

        # We open the file in write mode to stream results as they are ready
        with open(output_file, "w", encoding="utf-8") as out_f:
            for thread_idx, thread in enumerate(self.raw_data, start=1):
                print(f"\n--- Thread {thread_idx}/{len(self.raw_data)} ---")

                title = thread.get("title", "")
                url = thread.get("url", "")
                posts = thread.get("posts", [])

                # Normalize LaTeX in all post content before any LLM call
                posts = [
                    {**p, "content": normalize_latex(p.get("content", ""))}
                    for p in posts
                ]

                print("  Splitting tasks...")
                tasks = self._split_tasks(title, posts)
                print(f"  Found {len(tasks)} task(s)")

                for i, task in enumerate(tasks):
                    print(f"  Task {i + 1}/{len(tasks)}: filtering...")

                    question = normalize_latex(task["question"])
                    relevant_indices = task.get("post_indices", list(range(len(posts))))
                    has_inline = task.get("has_inline_solution", False)
                    relevant_posts = [p for p in posts if p["index"] in relevant_indices]

                    filt = self._filter_question(question, relevant_posts)

                    print("    -> Cleaning question LaTeX...")
                    question_clean = self._fix_latex_solution(question)

                    # Initialize the record
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

                    # Step A: Filter check
                    if not filt["keep"]:
                        print(f"    -> DISCARDED: {filt['reason']}")
                        self.stats["filtered_out"] += 1
                        self.processed_data.append(record)
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        continue

                    print("    -> KEPT. Classifying question...")
                    classification = self._classify_question(question)
                    category = classification.get("category")

                    # Step B: Category check
                    if category is None:
                        print("    -> Invalid or missing category! Discarding.")
                        self.stats["filtered_out"] += 1
                        self.processed_data.append(record)
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        continue

                    # Unpack enum to string safely
                    record["category"] = category.value if isinstance(category, Category) else category
                    record["category_reason"] = classification.get("reason")
                    print(f"    -> Category found: {record['category']}, reason: {record['category_reason']}")

                    print(f"    -> Finding answer (inline={has_inline})...")
                    raw_answer, answer_post_idx = self._find_correct_answer(
                        question, posts, relevant_indices, has_inline
                    )

                    # Step C: Answer check
                    if raw_answer is None:
                        print("    -> No answer found in thread. Discarding.")
                        self.stats["filtered_out"] += 1
                        self.processed_data.append(record)
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        continue

                    print("    -> Cleaning answer LaTeX...")
                    record["raw_answer"] = self._fix_latex_solution(normalize_latex(raw_answer))
                    record["answer_post_index"] = answer_post_idx

                    print("    -> Rewriting answer...")
                    solution = self._rewrite_answer(question_clean, raw_answer)

                    print("    -> Fixing solution LaTeX...")
                    record["solution"] = self._fix_latex_solution(solution)

                    print("    -> SUCCESS! Task fully processed and kept.")
                    self.stats["kept"] += 1
                    self.processed_data.append(record)

                    # Write final successful record to file
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("\n=== PIPELINE FINISHED ===")
        print(f"Results saved to: {output_file}")
        print("Pipeline Statistics:", json.dumps(self.stats, indent=2))
        save_dataset(output_file, self.dataset_dir)