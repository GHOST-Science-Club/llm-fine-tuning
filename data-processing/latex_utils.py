import re

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