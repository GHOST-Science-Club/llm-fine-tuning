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
import json
import os
import re
import sys
import glob
from pathlib import Path

import requests
from dotenv import load_dotenv

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

SPLIT_SYSTEM = """\
You are a math forum analyst. A forum thread may contain one or multiple distinct \
math problems. Your job is to identify each separate problem in the thread and \
list the post indices that are relevant to each problem.

Return ONLY a JSON array. Each element has:
  "question": the full problem statement (copy verbatim from the post, keep ALL LaTeX intact)
  "post_indices": list of integer post indices relevant to this problem
  "has_inline_solution": true if the same post that states the problem also contains a worked solution

IMPORTANT rules:
- Look for numbered examples inside a single post: markers like "Przykład 1", "Przykład 2",
  "Zadanie 1", "Example 1" each signal a separate task.
- When multiple examples live in one post (index N), set post_indices to [N] for each of them
  (plus any later posts that specifically discuss that example).
- If the post walks through the full solution immediately after the problem statement,
  set has_inline_solution to true.
- If the thread has only one problem, return a single-element array.

--- Example A: numbered examples inside one tutorial post ---
Thread title: "Sprzężenie – liczenie granic"
Posts:
  [0] author: nauczyciel — "\\text{Przykład 1} \\lim_{n\\to\\infty}(\\sqrt{n^2+2n}-n) \\text{ ...pełne rozwiązanie... Przykład 2} a_n = n^3-\\sqrt{n^6-5n^3} \\text{ ...pełne rozwiązanie...}"

Output:
[
  {
    "question": "\\lim_{n\\to\\infty}(\\sqrt{n^2+2n}-n)",
    "post_indices": [0],
    "has_inline_solution": true
  },
  {
    "question": "\\text{Oblicz } \\lim_{n\\to\\infty} a_n \\text{ gdzie } a_n = n^3-\\sqrt{n^6-5n^3}",
    "post_indices": [0],
    "has_inline_solution": true
  }
]

--- Example B: Q&A thread with one problem and discussion replies ---
Thread title: "Ciekawy iloczyn"
Posts:
  [0] author: mol_ksiazkowy — "\\text{Udowodnić, że } f(m)= \\frac{2}{3} (-1)^{m+1} m!^2 \\prod_{n=1}^m \\frac{n+m}{n^3+m^3}"
  [1] author: azanus111 — "\\text{Ustalmy } m \\text{, niech } f(m)= \\prod_{n \\neq m} \\frac{n-m}{n+m} \\cdot \\prod_{n \\neq m} \\frac{n^2+nm+m^2}{n^2-nm+m^2} \\text{ ...cnd}"
  [2] author: Jan Kraszewski — "\\text{No cóż, } (-1)^{m-1}=(-1)^{m+1}"

Output:
[
  {
    "question": "\\text{Udowodnić, że } f(m)= \\frac{2}{3} (-1)^{m+1} m!^2 \\prod_{n=1}^m \\frac{n+m}{n^3+m^3}",
    "post_indices": [0, 1, 2],
    "has_inline_solution": false
  }
]
"""


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

FILTER_SYSTEM = """\
You are a filter for a math fine-tuning dataset. Decide whether a math problem \
should be KEPT or DISCARDED.

Discard if the problem:
- Requires drawing, constructing, or sketching a figure (e.g. "naszkicuj", "skonstruuj", "narysuj")
- Is spam, off-topic, or not a math problem at all
- Is purely a meta-discussion (e.g. asking for a textbook recommendation)
- Cannot be answered without a visual/image that is attached to the post (contains_images: true
  AND the content references a figure, table, or drawing)
- Is only an incomplete fragment with no solvable question

Keep if the problem:
- Is a well-defined math problem (algebra, calculus, number theory, combinatorics, proofs, etc.)
- Can be solved using text and LaTeX notation only
- Is a tutorial post that states worked examples — keep each example as its own task

Respond with exactly two lines:
DECISION: YES   (or NO)
REASON: one short sentence

Few-shot examples:

--- Example 1 ---
Problem: \\text{Oblicz } \\lim_{n \\to \\infty} \\frac{n^2+1}{2n^2-3}
contains_images in relevant posts: false
DECISION: YES
REASON: Standard calculus limit problem, fully solvable in text.

--- Example 2 ---
Problem: \\text{Skonstruuj trójkąt o bokach 3, 4, 5 używając cyrkla i linijki i narysuj wszystkie wysokości.}
contains_images in relevant posts: false
DECISION: NO
REASON: Requires physical drawing/construction.

--- Example 3 ---
Problem: \\text{Hej, ktoś może polecić dobry podręcznik do analizy matematycznej?}
contains_images in relevant posts: false
DECISION: NO
REASON: Off-topic meta-discussion, not a math problem.

--- Example 4 ---
Problem: \\text{Udowodnij, że dla każdej liczby całkowitej } n \\text{, wyrażenie } n^2 + n \\text{ jest parzyste.}
contains_images in relevant posts: false
DECISION: YES
REASON: Proof problem solvable entirely in text.

--- Example 5 ---
Problem: \\text{Na rysunku poniżej dane są kąty trójkąta. Oblicz pole.}
contains_images in relevant posts: true
DECISION: NO
REASON: Problem depends on an attached image that cannot be read as text.
"""


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
# Step 2 — Find correct answer
# ---------------------------------------------------------------------------

FIND_ANSWER_SYSTEM = """\
You are reviewing a math forum thread. Given the problem and the list of posts, \
find the most complete and correct answer.

Rules:
- If the post that states the problem ALSO contains a full worked solution \
  (e.g. a tutorial post with "Przykład N … solution …"), extract that solution \
  directly from the problem post.
- Otherwise, look through the reply posts and pick the one with the most \
  complete and mathematically correct solution.
- Ignore posts that are pure meta-discussion (corrections about notation, arguments \
  about style) without actual math content.
- If no satisfactory answer exists anywhere, return exactly: NO_ANSWER

Return ONE line only:
POST_INDEX: <integer index of the post that contains the best answer>

--- Example A: inline solution in the problem post (tutorial thread) ---
Problem: \\lim_{n\\to\\infty}(\\sqrt{n^2+2n}-n)
Posts:
  [0] nauczyciel (contains_images=False): "\\text{Przykład 1} \\lim_{n\\to\\infty}(\\sqrt{n^2+2n}-n) \\text{ Niech } a=\\sqrt{n^2+2n} \\text{, } b=n \\text{. Korzystamy ze wzoru } a-b=\\frac{a^2-b^2}{a+b} \\text{:} =\\lim_{n\\to\\infty}\\frac{n^2+2n-n^2}{\\sqrt{n^2+2n}+n}=\\lim_{n\\to\\infty}\\frac{2n}{\\sqrt{n^2+2n}+n} \\text{ Dzielimy przez } n \\text{: } =\\frac{2}{\\sqrt{1+2/n}+1}\\to\\frac{2}{2}=1"

POST_INDEX: 0

--- Example B: answer in a reply post (Q&A thread) ---
Problem: \\text{Udowodnić, że } f(m)= \\frac{2}{3}(-1)^{m+1}m!^2 \\prod_{n=1}^m \\frac{n+m}{n^3+m^3}
Posts:
  [0] mol_ksiazkowy (contains_images=False): "\\text{Niech } f(m)= \\prod_{n \\neq m} \\frac{n^3-m^3}{n^3+m^3} \\text{ Udowodnić, że ...}"
  [1] azanus111 (contains_images=False): "\\text{Ustalmy } m \\text{, niech: } f(m)= \\prod_{n \\neq m} \\frac{n-m}{n+m} \\cdot \\prod_{n \\neq m} \\frac{n^2+nm+m^2}{n^2-nm+m^2} \\text{ ...cnd}"
  [2] Jan Kraszewski (contains_images=False): "\\text{No cóż, } (-1)^{m-1}=(-1)^{m+1}"

POST_INDEX: 1
"""


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
# Step 3 — Rewrite answer (chain-of-thought)
# ---------------------------------------------------------------------------

REWRITE_SYSTEM = """\
You are formatting a math forum answer into numbered steps. \
Your ONLY job is to split the answer into steps and fix LaTeX syntax. \
Do NOT add, remove, or change any mathematical content.

Rules:
- Split the answer into numbered steps: "Krok 1:", "Krok 2:", etc.
- Each step = one logical sentence or one formula from the original
- Copy all text and formulas EXACTLY — do not paraphrase, do not add explanations
- Fix LaTeX syntax only: use $...$ for inline math, \\[ ... \\] for display math
- End with \\textbf{Wynik:} and the final result in \\[ ... \\]
- If the original has no final numeric result, skip \\textbf{Wynik:}

--- Example ---
Raw answer:
  "\\text{Niech } a=\\sqrt{n^2+2n-1} \\text{, } b=n \\text{, korzystamy ze wzoru } a-b=\\frac{a^2-b^2}{a+b} \\text{, liczymy i wychodzi 1}"

Rewritten:

Krok 1: Niech $a=\\sqrt{n^2+2n-1}$, $b=n$, korzystamy ze wzoru $a-b=\\frac{a^2-b^2}{a+b}$.

Krok 2: Liczymy i wychodzi 1.

\\textbf{Wynik:}
\\[ \\lim_{n \\to \\infty}(\\sqrt{n^2+2n-1}-n) = 1 \\]
"""


def rewrite_answer(question: str, raw_answer: str) -> str:
    user_prompt = (
        f"Problem: {question[:1000]}\n\n"
        f"Raw answer: {raw_answer[:2000]}\n\n"
        "Rewritten solution:"
    )
    result = call_llm(REWRITE_SYSTEM, user_prompt).strip()
    debug("STEP 3 — rewrite_answer | full rewritten solution", result)
    return result

