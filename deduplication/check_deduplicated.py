"""
Inspection script — shows which training examples were removed during deduplication
and finds the most similar kept document.

Same output as the naive all-pairs version, but uses an inverted n-gram index so each
removed document is only compared against kept documents that share at least one n-gram.
Identical Jaccard scores and identical tie-breaking (lowest kept-doc index wins).
"""

import gzip, json, sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

removed_dir = Path("data/deduplicated/removed")
clean_dir   = Path("data/deduplicated/clean")
pairs_out   = Path("data/deduplicated/duplicate_pairs.jsonl")

NGRAM = 3


def word_ngrams(text: str, n: int = NGRAM) -> Counter:
    words = text.lower().split()
    return Counter(tuple(words[i:i+n]) for i in range(len(words) - n + 1))


def read_jsonl_dir(d: Path) -> list[dict]:
    docs = []
    for f in sorted(d.glob("*.jsonl*")):
        opener = gzip.open if f.suffix == ".gz" else open
        with opener(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                docs.append(json.loads(line))
    return docs


# ── load ─────────────────────────────────────────────────────────────────────
clean_docs = read_jsonl_dir(clean_dir)
removed_docs = read_jsonl_dir(removed_dir)
print(f"Loaded {len(clean_docs)} kept and {len(removed_docs)} removed documents", file=sys.stderr)

# ── build inverted index over kept documents ─────────────────────────────────
# ngram -> integer id, so postings can live in compact numpy arrays
ngram_ids: dict[tuple, int] = {}
post_ngram, post_doc, post_count = [], [], []
clean_sizes = np.zeros(len(clean_docs), dtype=np.int64)   # total n-grams per kept doc

for doc_idx, cd in enumerate(clean_docs):
    counts = word_ngrams(cd.get("text", ""))
    clean_sizes[doc_idx] = sum(counts.values())
    for ng, c in counts.items():
        nid = ngram_ids.get(ng)
        if nid is None:
            nid = len(ngram_ids)
            ngram_ids[ng] = nid
        post_ngram.append(nid)
        post_doc.append(doc_idx)
        post_count.append(c)

post_ngram = np.array(post_ngram, dtype=np.int32)
post_doc   = np.array(post_doc,   dtype=np.int32)
post_count = np.array(post_count, dtype=np.int32)

order = np.argsort(post_ngram, kind="stable")
post_ngram, post_doc, post_count = post_ngram[order], post_doc[order], post_count[order]
# start offset of every ngram id in the sorted postings
starts = np.searchsorted(post_ngram, np.arange(len(ngram_ids) + 1, dtype=np.int32))
print(f"Index: {len(ngram_ids)} distinct {NGRAM}-grams, {len(post_ngram)} postings", file=sys.stderr)

# ── for each removed doc, find the most similar kept doc ─────────────────────
with open(pairs_out, "w", encoding="utf-8") as pf:
    for doc in removed_docs:
        removed_text = doc.get("text", "")
        removed_ng = word_ngrams(removed_text)
        size_a = sum(removed_ng.values())

        docs_parts, mins_parts = [], []
        for ng, qc in removed_ng.items():
            nid = ngram_ids.get(ng)
            if nid is None:
                continue
            lo, hi = starts[nid], starts[nid + 1]
            docs_parts.append(post_doc[lo:hi])
            mins_parts.append(np.minimum(post_count[lo:hi], qc))

        best_score, best_doc = 0.0, None
        if docs_parts:
            cand_docs = np.concatenate(docs_parts)
            cand_mins = np.concatenate(mins_parts)
            # multiset intersection size per kept doc
            inter = np.bincount(cand_docs, weights=cand_mins, minlength=len(clean_docs))
            hit = np.flatnonzero(inter)
            # |A ∪ B| = |A| + |B| - |A ∩ B|  (multiset union = elementwise max)
            union = size_a + clean_sizes[hit] - inter[hit]
            scores = np.where(union > 0, inter[hit] / np.maximum(union, 1), 0.0)
            b = int(np.argmax(scores))          # first max wins -> lowest kept index
            if scores[b] > 0.0:
                best_score, best_doc = float(scores[b]), clean_docs[int(hit[b])]

        print("Removed (duplicate):", removed_text[:120])
        if best_doc:
            print(f"Most similar kept ({best_score:.0%}):", best_doc.get("text", "")[:120])
        else:
            print("Most similar kept: (none found)")
        print("---")

        pf.write(json.dumps({
            "removed_id": doc.get("id"),
            "removed_text": removed_text,
            "kept_id": (best_doc or {}).get("id"),
            "kept_text": (best_doc or {}).get("text"),
            "jaccard": round(best_score, 4),
            "removed_metadata": doc.get("metadata"),
        }, ensure_ascii=False) + "\n")

print(f"Wrote removed->kept pairs to {pairs_out}", file=sys.stderr)
