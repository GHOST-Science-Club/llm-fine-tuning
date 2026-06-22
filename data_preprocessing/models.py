import asyncio
import json
from contextlib import ExitStack
from enum import Enum
from pathlib import Path

from datasets import load_dataset

from .utils import debug, save_dataset, LLMClient
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

    def __init__(self, input_source: Path | str, output_file: Path, dataset_destination: Path | str, checkpoint_file: Path, llm: LLMClient, batch_size: int, log_file: Path | None = None, quiet: bool = False):
        self.input_source = input_source
        self.output_file = output_file
        self.log_file = log_file
        self.dataset_destination = dataset_destination
        self.checkpoint_file = checkpoint_file
        self.llm = llm
        self.batch_size = batch_size
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
    async def _split_tasks(self, title: str, posts: list[dict]) -> list[dict]:
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


        raw = await self.llm.call(SPLIT_SYSTEM, user_prompt)
        debug("STEP 0 — split_tasks | raw LLM output", raw)

        try:
            clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            tasks = json.loads(clean)

            if isinstance(tasks, list) and tasks:
                debug("STEP 0 — split_tasks | parsed tasks", json.dumps(tasks, ensure_ascii=False, indent=2))
                return tasks

        except (json.JSONDecodeError, KeyError):
            debug("STEP 0 — split_tasks | JSON parse failed, using fallback", raw)
            self.stats["llm_parse_errors"] +=  1

        return [{
            "question": posts[0]["content"] if posts else title,
            "post_indices": list(range(len(posts)))
        }]

    async def _filter_question(self, question: str, relevant_posts: list[dict] | None = None) -> dict:
        """Returns {"keep": bool, "reason": str}."""
        has_images = any(p.get("contains_images", False) for p in (relevant_posts or []))
        user_prompt = (
            f"Problem: {question[:1000]}\n"
            f"contains_images in relevant posts: {str(has_images).lower()}"
        )
        raw = await self.llm.call(FILTER_SYSTEM, user_prompt)
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

    async def _classify_question(self, question: str) -> dict:
        """
        Assign category to the question based on the solution
        (exact value, expression, proof, complex)
        Returns {"category": Category | None, "reason": str}.
        """
        user_prompt = f"Problem: {question[:1000]}"
        raw = await self.llm.call(CLASSIFY_SYSTEM, user_prompt)

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
            self.stats["classification_errors"] += 1
            debug("STEP 2", f"WARNING: Invalid category from LLM: {category_str}")

        return {"category": category, "reason": reason}

    async def _find_correct_answer(self, question: str, posts: list[dict], relevant_indices: list[int],
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

        raw = (await self.llm.call(FIND_ANSWER_SYSTEM, user_prompt)).strip()
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

    async def _rewrite_answer(self, question: str, raw_answer: str) -> str:
        user_prompt = (
            f"Problem: {question[:1000]}\n\n"
            f"Raw answer: {raw_answer[:2000]}\n\n"
            "Rewritten solution:"
        )
        result = (await self.llm.call(REWRITE_SYSTEM, user_prompt)).strip()
        debug("STEP 3 — rewrite_answer | full rewritten solution", result)
        return result

    async def _fix_latex(self, text: str) -> str:
        """LLM pass to catch any remaining LaTeX syntax errors in the given text."""
        result = (await self.llm.call(FIX_LATEX_SYSTEM, text)).strip()
        # Strip markdown code fences in case the model wraps its output
        result = result.removeprefix('```latex').removeprefix('```').removesuffix('```').strip()
        debug("STEP 4 — fix_latex | corrected output", result)
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

    async def _process_task(self, task: dict, posts: list[dict], url: str, title: str,
                            label: str) -> tuple[dict | None, dict | None]:
        """
        Run a single task through the full LLM chain.

        Returns (full_record, clean_record):
          - (None, None)         → skipped before any record could be built
          - (full_record, None)  → processed but not usable (filtered / no answer / error)
          - (full_record, clean) → success

        Records are returned, never written here — the caller owns all file I/O so
        concurrently running tasks can't interleave their writes.
        """
        try:
            question = normalize_latex(task["question"])
            relevant_indices = task.get("post_indices", list(range(len(posts))))
            has_inline = task.get("has_inline_solution", False)
        except (KeyError, TypeError, AttributeError) as e:
            print(f"  {label} -> DISCARDED: Distorted task structure from LLM. Skipping... ({e})")
            self.stats["llm_parse_errors"]+=  1
            return None, None

        relevant_posts = [p for p in posts if p["index"] in relevant_indices]
        try:
            filt = await self._filter_question(question, relevant_posts)
        except Exception as e:

            self.stats["llm_parse_errors"] +=  1
            print(f"  {label} Error while filtering question! {e}")
            return None, None
        try:
            question_clean = await self._fix_latex(question)
        except Exception as e:
            self.stats["llm_parse_errors"] += 1
            print(f"  {label} Error while cleaning question! {e}")
            return None, None

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
                print(f"  {label} -> DISCARDED: {filt['reason']}")
            self.stats["filtered_out"] += 1
        else:
            try:
                classification = await self._classify_question(question_clean)
                category = classification.get("category", None)
                category_reason = classification.get("reason", "")
            except Exception as e:
                print(f"  {label} Error while classifying question! {e}")
                self.stats["classification_errors"] += 1
                category = None

            # Step B: Category check
            if category is None:
                if not self.quiet:
                    print(f"  {label} -> Invalid or missing category! Discarding.")
                self.stats["filtered_out"] += 1
            else:
                try:
                    raw_answer, answer_post_idx = await self._find_correct_answer(
                        question, posts, relevant_indices, has_inline
                    )
                except Exception as e:
                    print(f"  {label} Error while looking for correct answer! {e}")
                    raw_answer = None
                    answer_post_idx = None

                # Step C: Answer check
                if raw_answer is None:
                    if not self.quiet:
                        print(f"  {label} -> No answer found in thread. Discarding.")
                    self.stats["filtered_out"] += 1
                else:
                    try:
                        raw_answer_clean = await self._fix_latex(normalize_latex(raw_answer))
                        rewritten = await self._rewrite_answer(question_clean, raw_answer_clean)
                        solution = await self._fix_latex(rewritten)
                        self.stats["kept"] += 1
                        success = True
                        if not self.quiet:
                            print(f"  {label} -> SUCCESS! Task fully processed and kept.")
                    except Exception as e:
                        print(f"  {label} Error during final rewriting/cleaning steps! {e}")

        # Full diagnostic record is always returned; the clean record only on success.
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
        clean_record = self._build_clean_record(full_record) if success else None
        return full_record, clean_record

    async def _process_thread(self, thread_idx: int, thread: dict) -> list[tuple[dict | None, dict | None]]:
        """Split one thread into tasks and process all of them concurrently."""
        if not self.quiet:
            print(f"\n--- Thread {thread_idx}/{len(self.raw_data)} ---")

        title = thread.get("title", "")
        url = thread.get("url", "")
        posts = thread.get("posts", [])

        # Normalize LaTeX in all post content before any LLM call.
        posts = [
            {**p, "content": normalize_latex(p.get("content", ""))}
            for p in posts
        ]

        try:
            tasks = await self._split_tasks(title, posts)
        except Exception as e:
            print(f"Error while splitting tasks (thread {thread_idx})! {e}")
            return []

        if not self.quiet:
            print(f"  Thread {thread_idx}: found {len(tasks)} task(s)")

        results = await asyncio.gather(
            *(
                self._process_task(task, posts, url, title, f"t{thread_idx} #{i + 1}/{len(tasks)}")
                for i, task in enumerate(tasks)
            ),
            return_exceptions=True,
        )

        records: list[tuple[dict | None, dict | None]] = []
        for r in results:
            if isinstance(r, Exception):
                print(f"Unexpected error processing a task in thread {thread_idx}: {r}")
                self.stats["llm_parse_errors"] += 1
                continue
            records.append(r)
        return records

    async def run(self) -> None:
        """
        Main pipeline method.
        Processes loaded data concurrently in batches of threads, applies LLM
        transformations/filters, and saves the output.
        """
        # 1. Load the raw data into self.raw_data
        self._load_data()

        # Checkpoint is thread-level: "<next_thread_idx>:0". We only advance it once
        # a whole batch is done, so resuming never re-emits already-written records.
        start_thread_idx = 0
        if self.checkpoint_file.is_file():
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as checkpoint_f:
                    start_thread_idx = int(checkpoint_f.read().strip().split(":")[0])
            except Exception:
                print("Warning! Checkpoint could not have been found!")
        start_thread_idx = max(0, start_thread_idx)

        # append mode if starting from checkpoint
        file_mode = "a" if start_thread_idx > 0 else "w"

        total = len(self.raw_data)
        if not self.quiet:
            print(f"\nStarting to process {total - start_thread_idx} threads "
                  f"(batch size {self.batch_size})...")
            print(f"Starting from thread index {start_thread_idx}")

        # Clean records → out_f, full diagnostic records → log_f. The log file is
        # optional. ExitStack closes whatever was opened even if the loop raises.
        with ExitStack() as stack:
            out_f = stack.enter_context(open(self.output_file, file_mode, encoding="utf-8"))
            log_f = stack.enter_context(open(self.log_file, file_mode, encoding="utf-8")) if self.log_file else None

            for batch_start in range(start_thread_idx, total, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total)
                batch = self.raw_data[batch_start:batch_end]

                # Process every thread in the batch concurrently; the LLMClient's
                # semaphore caps how many requests are actually in flight at once.
                thread_results = await asyncio.gather(
                    *(
                        self._process_thread(idx, thread)
                        for idx, thread in enumerate(batch, start=batch_start)
                    ),
                    return_exceptions=True,
                )

                # Write the whole batch in thread order, then advance the checkpoint.
                for idx, tr in zip(range(batch_start, batch_end), thread_results):
                    if isinstance(tr, Exception):
                        print(f"Unexpected error processing thread {idx}: {tr}")
                        continue
                    for full_record, clean_record in tr:
                        if full_record and log_f:
                            log_f.write(json.dumps(full_record, ensure_ascii=False) + "\n")
                        if clean_record:
                            out_f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")

                out_f.flush()
                if log_f:
                    log_f.flush()

                try:
                    with open(self.checkpoint_file, "w", encoding="utf-8") as checkpoint_f:
                        checkpoint_f.write(f"{batch_end}:0")
                except Exception:
                    print("Warning! Checkpoint could not have been saved!")

        print("\n=== PIPELINE FINISHED ===")
        print(f"Clean records saved to: {self.output_file}")
        if self.log_file:
            print(f"Full diagnostic logs saved to: {self.log_file}")
        print("Pipeline Statistics:", json.dumps(self.stats, indent=2))
        save_dataset(self.output_file, self.dataset_destination)