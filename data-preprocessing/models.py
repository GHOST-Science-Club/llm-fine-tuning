import json
from contextlib import ExitStack
from enum import Enum
from pathlib import Path

from datasets import load_dataset

from .utils import debug, call_llm, save_dataset
from .prompts import SPLIT_SYSTEM, FILTER_SYSTEM, FIND_ANSWER_SYSTEM, CLASSIFY_SYSTEM, REWRITE_SYSTEM, FIX_LATEX_SYSTEM
from .latex_utils import normalize_latex

class Category(str, Enum):
    EXACT_VALUE = "EXACT_VALUE"
    EXPRESSION = "EXPRESSION"
    PROOF = "PROOF"
    COMPLEX = "COMPLEX"

# Fields that make up the clean, training-ready dataset (everything else is diagnostic).
CLEAN_FIELDS = ("source_url", "question", "category", "solution")

class DataProcessingPipeline:

    def __init__(self, input_source: Path | str, output_file: Path, dataset_destination: Path | str, checkpoint_file: Path, log_file: Path | None = None, quiet: bool = False):
        self.input_source = input_source
        self.output_file = output_file
        self.log_file = log_file
        self.dataset_destination = dataset_destination
        self.checkpoint_file = checkpoint_file
        self.raw_data = []
        self.quiet = quiet
        self.stats = {
            "loaded": 0,
            "filtered_out": 0,
            "kept": 0,
            "llm_parse_errors": 0,
            "classification_errors": 0
        }


    def _load_data(self) -> None:
        """Loading raw data from (HF Hub or local JSONL."""
        debug("Loading data", f"Attempting to load dataset from: {self.input_source}")

        try:
            # Check if input_source is a local file path or a HF Hub repo identifier
            if isinstance(self.input_source, Path) and self.input_source.suffix == '.jsonl':
                dataset = load_dataset("json", data_files={"train": str(self.input_source)}, split="train")
            elif isinstance(self.input_source, str):
                dataset = load_dataset(self.input_source, split='train')
            else:
                raise ValueError("Invalid input source. Must be a local JSONL file path or a Hugging Face Hub dataset identifier.")

            loaded_records = dataset.to_list()
            self.raw_data.extend(loaded_records)
            self.stats["loaded"] += len(loaded_records)

            debug("Loading data", f"Successfully loaded {len(loaded_records)} records.")

        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from {self.input_source}. Details: {e}")
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

    @staticmethod
    def _build_full_record(
        url: str,
        title: str,
        question: str,
        filt: dict,
        category: "Category | str | None" = None,
        category_reason: str | None = None,
        raw_answer: str | None = None,
        answer_post_index: int | None = None,
        solution: str | None = None,
    ) -> dict:
        """
        Build the full diagnostic record: question + all metadata, filtering and
        classification results. Written to the local log file for inspection.
        """
        return {
            "source_url": url,
            "source_title": title,
            "question": question,
            "kept": filt["keep"],
            "filter_reason": filt["reason"],
            "category": category.value if isinstance(category, Category) else category,
            "category_reason": category_reason,
            "raw_answer": raw_answer,
            "answer_post_index": answer_post_index,
            "solution": solution,
        }

    @staticmethod
    def _build_clean_record(full_record: dict) -> dict:
        """Project a full record down to the clean, training-ready fields only."""
        return {key: full_record[key] for key in CLEAN_FIELDS}

    def run(self) -> None:
        """
        Main pipeline method.
        Processes loaded data, applies LLM transformations/filters, and saves the output.
        """
        # 1. Load the raw data into self.raw_data
        self._load_data()


        start_thread_idx, start_task_idx = 0, 0
        if self.checkpoint_file.is_file():
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as checkpoint_f:
                    start_thread_idx, start_task_idx = checkpoint_f.read().strip().split(":")
                    start_thread_idx, start_task_idx = int(start_thread_idx), int(start_task_idx)
            except Exception as e:
                print("Warning! Checkpoint could not have been found!")

        # append mode if starting from checkpoint
        file_mode = "a" if start_thread_idx > 0 or start_task_idx > 0 else "w"

        start_thread_idx = max(0, start_thread_idx)
        start_task_idx = max(0, start_task_idx)
        if not self.quiet:
            print(f"\nStarting to process {len(self.raw_data) - start_thread_idx} threads...")
            print(f"\nStarting from the {start_thread_idx} index")

        # Stream results as they are ready: clean records → out_f, full diagnostic records → log_f.
        # The log file is optional — when self.log_file is None, full records are simply not written.
        # ExitStack closes whatever was opened (both files, or just out_f) even if the loop raises.
        with ExitStack() as stack:
            out_f = stack.enter_context(open(self.output_file, file_mode, encoding="utf-8"))
            log_f = stack.enter_context(open(self.log_file, file_mode, encoding="utf-8")) if self.log_file else None
            for thread_idx, thread in enumerate(self.raw_data[start_thread_idx:], start=start_thread_idx):
                if not self.quiet:
                    print(f"\n--- Thread {thread_idx}/{len(self.raw_data)} ---")

                title = thread.get("title", "")
                url = thread.get("url", "")
                posts = thread.get("posts", [])

                # Normalize LaTeX in all post content before any LLM call
                posts = [
                    {**p, "content": normalize_latex(p.get("content", ""))}
                    for p in posts
                ]
                try:
                    if not self.quiet:
                        print("  Splitting tasks...")
                    tasks = self._split_tasks(title, posts)
                    if not self.quiet:
                        print(f"  Found {len(tasks)} task(s)")
                except Exception as e:
                    print(f"Error while splitting tasks! {e}")
                    continue

                for i, task in enumerate(tasks):

                    if thread_idx == start_thread_idx and i < start_task_idx:
                        if not self.quiet:
                            print(f"  Task {i + 1}/{len(tasks)}: I'm skipping (already done)...")
                        continue
                    if not self.quiet:
                        print(f"  Task {i + 1}/{len(tasks)}: filtering...")
                    try:
                        question = normalize_latex(task["question"])
                        relevant_indices = task.get("post_indices", list(range(len(posts))))
                        has_inline = task.get("has_inline_solution", False)
                    except (KeyError, TypeError, AttributeError) as e:
                        print(f"    -> DISCARDED: Distorted task structure from LLM. Skipping... ({e})")
                        self.stats["llm_parse_errors"] = self.stats.get("llm_parse_errors", 0) + 1
                        continue
                    relevant_posts = [p for p in posts if p["index"] in relevant_indices]
                    try:
                        filt = self._filter_question(question, relevant_posts)
                    except Exception as e:
                        print(f"Error while filtering questions! {e}")
                        continue
                    try:
                        if not self.quiet:
                            print("    -> Cleaning question LaTeX...")
                        question_clean = self._fix_latex_solution(question)
                    except Exception as e:
                        print(f"Error while cleaning question! {e}")
                        continue

                    # Accumulate fields as the task progresses through the steps.
                    category = None
                    category_reason = None
                    raw_answer_clean = None
                    answer_post_idx = None
                    solution = None
                    success = False

                    # Step A: Filter check
                    if not filt["keep"]:
                        if not self.quiet:
                            print(f"    -> DISCARDED: {filt['reason']}")
                        self.stats["filtered_out"] += 1
                    else:
                        try:
                            if not self.quiet:
                                print("    -> KEPT. Classifying question...")
                            classification = self._classify_question(question_clean)
                            category = classification.get("category", None)
                            category_reason = classification.get("reason", "")
                        except Exception as e:
                            print(f"Error while classifying question! {e}")
                            self.stats["classification_errors"] += 1
                            category = None

                        # Step B: Category check
                        if category is None:
                            if not self.quiet:
                                print("    -> Invalid or missing category! Discarding.")
                            self.stats["filtered_out"] += 1
                        else:
                            if not self.quiet:
                                cat_str = category.value if isinstance(category, Category) else category
                                print(f"    -> Category found: {cat_str}, reason: {category_reason}")
                            try:
                                if not self.quiet:
                                    print(f"    -> Finding answer (inline={has_inline})...")
                                raw_answer, answer_post_idx = self._find_correct_answer(
                                    question, posts, relevant_indices, has_inline
                                )
                            except Exception as e:
                                print(f"Error while looking for correct answer! {e}")
                                raw_answer = None
                                answer_post_idx = None


                            # Step C: Answer check
                            if raw_answer is None:
                                if not self.quiet:
                                    print("    -> No answer found in thread. Discarding.")
                                self.stats["filtered_out"] += 1
                            else:
                                try:
                                    if not self.quiet:
                                        print("    -> Cleaning answer LaTeX...")
                                    raw_answer_clean = self._fix_latex_solution(normalize_latex(raw_answer))

                                    if not self.quiet:
                                        print("    -> Rewriting answer...")
                                    rewritten = self._rewrite_answer(question_clean, raw_answer_clean)

                                    if not self.quiet:
                                        print("    -> Fixing solution LaTeX...")
                                    solution = self._fix_latex_solution(rewritten)

                                    if not self.quiet:
                                        print("    -> SUCCESS! Task fully processed and kept.")
                                    self.stats["kept"] += 1
                                    success = True
                                except Exception as e:
                                    print(f"Error during final rewriting/cleaning steps! {e}")

                    # Build records once from the accumulated fields.
                    # Full diagnostic record is always logged; the clean record only on success.
                    full_record = self._build_full_record(
                        url=url,
                        title=title,
                        question=question_clean,
                        filt=filt,
                        category=category,
                        category_reason=category_reason,
                        raw_answer=raw_answer_clean,
                        answer_post_index=answer_post_idx,
                        solution=solution,
                    )
                    if log_f:
                        log_f.write(json.dumps(full_record, ensure_ascii=False) + "\n")
                    if success:
                        clean_record = self._build_clean_record(full_record)
                        out_f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")

                    try:
                        with open(self.checkpoint_file, "w", encoding="utf-8") as checkpoint_f:
                            checkpoint_f.write(f"{thread_idx}:{i+1}")
                    except Exception as e:
                        print("Warning! Checkpoint could not have been saved!")

                start_task_idx = 0
                try:
                    with open(self.checkpoint_file, "w", encoding="utf-8") as checkpoint_f:
                        checkpoint_f.write(f"{thread_idx + 1}:0")
                except FileNotFoundError:
                    print("Warning! Checkpoint could not have been found!")

        print("\n=== PIPELINE FINISHED ===")
        print(f"Clean records saved to: {self.output_file}")
        if self.log_file:
            print(f"Full diagnostic logs saved to: {self.log_file}")
        print("Pipeline Statistics:", json.dumps(self.stats, indent=2))
        save_dataset(self.output_file, self.dataset_destination)