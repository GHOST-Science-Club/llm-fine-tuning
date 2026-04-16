import json
from utils import debug, call_llm
from prompts import SPLIT_SYSTEM, FILTER_SYSTEM, FIX_LATEX_SYSTEM, FIND_ANSWER_SYSTEM, REWRITE_SYSTEM, CLASSIFY_SYSTEM
from models import Category

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

    raw = call_llm(SPLIT_SYSTEM, user_prompt)
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

# ---------------------------------------------------------------------------
# Step 2 - Assign category to the question based on the solution
# (exact value, expression, proof, complex)
# ----------------------------------------------------------------------------

def classify_question(question: str) -> dict:
    """Returns {"category": Category | None, "reason": str}."""
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
        debug("STEP 2", f"WARNING: Invalid category from LLM: {category_str}")

    return {"category": category, "reason": reason}

# ---------------------------------------------------------------------------
# Step 3 — Find correct answer
# ---------------------------------------------------------------------------

def find_answer(question: str, posts: list[dict], relevant_indices: list[int],
                has_inline_solution: bool = False,) -> tuple[str | None, int | None]:
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



# ---------------------------------------------------------------------------
# Step 4 — Rewrite answer (chain-of-thought)
# ---------------------------------------------------------------------------

def rewrite_answer(question: str, raw_answer: str) -> str:
    user_prompt = (
        f"Problem: {question[:1000]}\n\n"
        f"Raw answer: {raw_answer[:2000]}\n\n"
        "Rewritten solution:"
    )
    result = call_llm(REWRITE_SYSTEM, user_prompt).strip()
    debug("STEP 3 — rewrite_answer | full rewritten solution", result)
    return result

# ---------------------------------------------------------------------------
# Step 5 — Fix LaTeX (final LLM pass)
# ---------------------------------------------------------------------------

def fix_latex_solution(solution: str) -> str:
    """Final LLM pass to catch any remaining LaTeX syntax errors in the solution."""
    result = call_llm(FIX_LATEX_SYSTEM, solution).strip()
    # Strip markdown code fences in case the model wraps its output
    result = result.removeprefix('```latex').removeprefix('```').removesuffix('```').strip()
    debug("STEP 4 — fix_latex_solution | corrected output", result)
    return result
