# Data Cleaning

Two-step pipeline for cleaning training datasets before LLM fine-tuning:
1. **Decontamination** — remove training examples that appear in evaluation benchmarks
2. **Deduplication** — remove near-duplicate training examples

Both steps read HF datasets (Arrow format) or JSONL files and produce a clean HF dataset as output.

| | Decontamination | Deduplication |
|---|---|---|
| **Removes** | Training examples that appear in benchmarks | Training examples that are duplicates of each other |
| **Catches paraphrases?** | No — requires exact n-gram match | Yes — compares overall similarity via MinHash |
| **Typical use** | Run once against the benchmark before training | Run to deduplicate the training set itself |

---

## Files

| File | Purpose |
|------|---------|
| `readers.py` | Shared utilities — reads HF datasets and JSONL files into datatrove Documents. Used by all other scripts. |
| `decontaminate.py` | Main decontamination script. Builds an n-gram hash index from benchmarks, then filters out training examples that share any n-gram with a benchmark task or solution. Catches verbatim copying only — does not catch paraphrases. |
| `deduplicate.py` | Main deduplication script. Runs a 4-stage MinHash LSH pipeline to find and remove near-duplicate training examples. Estimates overall text similarity so it can catch paraphrases, not just exact copies. |
| `load_data.py` | Preview script — loads datasets and prints sample documents. Useful for checking that column names and text are loading correctly before running the full pipeline. |
| `check_contaminated.py` | Inspection script — shows which training examples were removed during decontamination and which benchmark task they matched. |
| `check_deduplicated.py` | Inspection script — shows which training examples were removed during deduplication and finds the most similar kept document. |
| `check_hf_content.py` | Inspection script — prints every column of an HF dataset row by row. Useful for verifying the output dataset structure. |
| `requirements.txt` | Python dependencies (`datatrove[io]`, `datasets`). |

---

## Key concepts

### N-gram
A sequence of N consecutive words from a text. For example, with N=3, the sentence "the cat sat" produces one 3-gram: `["the", "cat", "sat"]`. Used in both decontamination and deduplication to represent text as a set of overlapping chunks.

### Decontamination — how it works
1. For each benchmark task, all n-grams from the `task` and `solution` fields are hashed and saved to a binary index file (one file per task).
2. Each training example has its `question` and `raw_answer` fields concatenated and split into n-grams.
3. If any n-gram from the training example appears in any benchmark index file, the example is removed.

### MinHash
A technique for estimating how similar two documents are without comparing them word by word. Each document is represented as a fixed-size array of hash values (a "signature"). Documents with similar content will tend to have similar signatures. Computing similarity between signatures is much faster than comparing full texts.

### LSH (Locality-Sensitive Hashing)
A method for finding similar documents efficiently at scale. The MinHash signature is split into `num_buckets` bands of `hashes_per_bucket` values each. Two documents are considered duplicate candidates if their signatures match completely within **at least one** band. This avoids comparing every document against every other document (which would be O(n²)).

### Deduplication — how it works (4 stages)
1. **Signature** — compute a MinHash signature for every training document.
2. **Buckets** — group documents into LSH buckets; documents landing in the same bucket are duplicate candidates.
3. **Cluster** — use union-find to merge overlapping candidate pairs into clusters; mark all but one document per cluster for removal.
4. **Filter** — read training data again; forward kept documents to the clean output, write removed ones to the removed output.

---

## Usage

### Decontamination

```bash
# default paths (data/dataset/processed + data/benchmark.jsonl)
python decontamination/decontaminate.py

# custom paths and multiple benchmarks
python decontamination/decontaminate.py \
  --datasets data/dataset/processed \
  --benchmarks data/benchmark.jsonl data/benchmark2.jsonl

# inspect removed examples
python decontamination/check_contaminated.py
```

Output: `data/decontaminated/hf_dataset/` (clean), `data/decontaminated/removed/` (flagged)

### Deduplication

```bash
# default paths (data/dataset/processed)
python deduplication/deduplicate.py

# run on a JSONL file for testing
python deduplication/deduplicate.py \
  --datasets data/benchmark.jsonl \
  --text-key task solution

# inspect removed examples
python deduplication/check_deduplicated.py
```

Output: `data/deduplicated/hf_dataset/` (clean), `data/deduplicated/removed/` (duplicates)

---

## Tuning decontamination sensitivity

Controlled by `--ngram-size` in `decontaminate.py` (default: **9**).

An n-gram is a sequence of N consecutive words. The filter removes a training example if any of its n-grams appear in the benchmark index.

| `--ngram-size` | Effect |
|----------------|--------|
| **5–7** | More sensitive — flags examples sharing shorter phrases with benchmarks |
| **9** (default) | Balanced — datatrove's recommended default |
| **12–15** | Less sensitive — only flags near-identical passages |

Smaller values increase recall (catch more contamination) but also increase false positives. The benchmark index is built from both `task` and `solution` fields, so any training example whose `question` or `raw_answer` shares an n-gram with either field is removed.

---

## Tuning deduplication sensitivity

Controlled by three parameters in `deduplicate.py`:

### `--ngram-size` (default: 3)
Size of word shingles used to represent documents as sets of tokens.

- Smaller (2–3): more sensitive, catches paraphrases with different wording but same equations
- Larger (5–8): only catches documents with long verbatim overlaps

### `--hashes-per-bucket` (default: 3)
Number of MinHash values that must **all** match within one LSH band for a pair to be flagged as duplicate.

- Lower (1–2): very sensitive — any small overlap triggers deduplication
- Higher (6–8): conservative — requires most of the document to be identical

### `--num-buckets` (default: 14)
Number of independent LSH bands. A document pair is flagged if it matches in **at least one** band.

- More buckets: better recall at the same `hashes-per-bucket` threshold
- Must be a divisor of `--tasks`

### Quick reference

| Goal | Setting |
|------|---------|
| Catch exact copies only | `--ngram-size 5 --hashes-per-bucket 8` |
| Catch paraphrases (default) | `--ngram-size 3 --hashes-per-bucket 3` |
| Aggressive — catch loose rewrites | `--ngram-size 2 --hashes-per-bucket 1` |


experimentally the best working parameters are:

`python deduplication/deduplicate.py --datasets data/benchmark.jsonl --text-key task solution --ngram-size 3 --hashes-per-bucket 2`

they allowed for deletion of exact matches, paraphrased matches, but nothing more

remember to remove folder with deduplicated data before running script

`Remove-Item -Recurse -Force data/deduplicated`



---

## Note on `--tasks` and `--num-buckets`

MinHash stage 2 requires `--tasks` to be divisible by `--num-buckets`. The script uses `start_method="spawn"` which works on both **Windows and Linux**, so multiple tasks are supported on all platforms.

Default is `--tasks 14 --num-buckets 14`. More buckets means more chances for a duplicate pair to be detected (better recall).

To run single-threaded (e.g. for debugging):

```bash
python deduplication/deduplicate.py --tasks 1 --num-buckets 1
```


### testing. if you wish to test it as i did you can put data floder from here https://drive.google.com/file/d/1lJH0J5R2QEa_5Os_ffndWd4cLh3N5sE_/view?usp=sharing and put it in the repo unipped
