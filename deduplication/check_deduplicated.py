import gzip, json, sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

removed_dir = Path("data/deduplicated/removed")
clean_dir   = Path("data/deduplicated/clean")


def word_ngrams(text: str, n: int = 3) -> Counter:
    words = text.lower().split()
    return Counter(tuple(words[i:i+n]) for i in range(len(words) - n + 1))


def jaccard(a: Counter, b: Counter) -> float:
    intersection = sum((a & b).values())
    union = sum((a | b).values())
    return intersection / union if union else 0.0


# Load all kept documents into memory
clean_docs = []
for f in sorted(clean_dir.glob("*.jsonl*")):
    with gzip.open(f, "rt", encoding="utf-8") as fh:
        for line in fh:
            clean_docs.append(json.loads(line))

# For each removed duplicate, find the most similar kept doc
for f in sorted(removed_dir.glob("*.jsonl*")):
    with gzip.open(f, "rt", encoding="utf-8") as fh:
        for line in fh:
            doc = json.loads(line)
            removed_text = doc.get("text", "")
            removed_ng = word_ngrams(removed_text)

            best_score, best_doc = 0.0, None
            for cd in clean_docs:
                score = jaccard(removed_ng, word_ngrams(cd.get("text", "")))
                if score > best_score:
                    best_score, best_doc = score, cd

            print("Removed (duplicate):", removed_text[:120])
            if best_doc:
                print(f"Most similar kept ({best_score:.0%}):", best_doc.get("text", "")[:120])
            else:
                print("Most similar kept: (none found)")
            print("---")
