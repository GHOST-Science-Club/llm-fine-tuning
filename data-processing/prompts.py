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

FIX_LATEX_SYSTEM = """\
You are a LaTeX editor. Convert the given text into clean, compilable LaTeX \
that can be pasted directly into Overleaf or KaTeX.

Step 1 — Fix implicit-math-mode format:
Some inputs are written as a single implicit math block where Polish words are inside
\\text{} and math expressions appear between them without delimiters, like:
  \\text{Niech } a=\\sqrt{n^2+2n-1} \\text{, korzystamy ze wzoru } a-b=\\frac{a^2-b^2}{a+b}
Convert this to proper LaTeX where plain text is outside math and only math is in $...$:
  Niech $a=\\sqrt{n^2+2n-1}$, korzystamy ze wzoru $a-b=\\frac{a^2-b^2}{a+b}$

Step 2 — Fix remaining syntax errors:
- \\displaystyle{expr}  →  \\[expr\\]
- ^{}  →  remove empty superscripts
- bare ...  →  \\cdots  (inside math) or \\ldots  (in text)
- **text**  →  \\textbf{text}
- \\infty as list terminator  →  \\cdots
- spaces in index braces: _{  x  }  →  _{x}

--- Example ---
Input:
  \\text{Niech } f(m)=\\prod_{n \\neq m} \\frac{n^3-m^3}{n^3+m^3} \\text{ Udowodnić, że } f(m)=\\frac{2}{3}(-1)^{m+1}

Output:
  Niech $f(m)=\\prod_{n \\neq m} \\frac{n^3-m^3}{n^3+m^3}$. Udowodnić, że $f(m)=\\frac{2}{3}(-1)^{m+1}$

Return ONLY the corrected LaTeX. No explanations, no markdown code fences.
"""