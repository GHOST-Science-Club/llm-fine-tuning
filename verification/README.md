# `verification/` — independent correctness check of the preprocessed dataset

`data_preprocessing/` grades its own output with the same model that produced it, so its
`VALID` verdicts are not an independent quality signal. This module takes the finished
JSONL, samples N records at random, and asks a stronger third-party model (OpenAI) to judge
each one. It is **read-only** with respect to the dataset — it reports, it does not filter.

## Setup

```bash
pip install -r verification/requirements.txt
cp verification/.env.example verification/.env   # then fill in OPENAI_API_KEY
```

Configuration is read from `verification/.env` (loaded with `override=True`, so `.env` wins
over the ambient environment).

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(empty)* | API key; the run aborts immediately if it is unset |
| `OPENAI_BASE_URL` | *(unset)* | Override the endpoint; leave unset for the official API |
| `MODEL` | `gpt-5-mini` | Judge model. Cheap reasoning model — better at catching real math errors than a non-reasoning one |
| `NUM_EXAMPLES` | `30` | How many records to sample |
| `SEED` | `42` | Sampling seed; the same seed picks the same examples |
| `MAX_CONCURRENCY` | `5` | Cap on concurrent in-flight requests |
| `DEBUG` | `false` | `true` prints the full solution from the dataset and the raw model reply for every example |

## Running

Run as a module from the repository root:

```bash
python -m verification.verify                 # uses the .env settings
python -m verification.verify -n 50           # check 50 examples
python -m verification.verify -m gpt-5 -s 7   # different model and sample
```

Flags: `-n/--num-examples`, `-m/--model`, `-s/--seed`, `-i/--input`, `-o/--output`. Each one
defaults to the corresponding `.env` value. The input defaults to
`data/2026_08_14_data_preprocessing_output.jsonl` at the repo root.

Note that `TEMPERATURE` is deliberately not sent — reasoning models reject it, and omitting
it keeps the module working with both reasoning and non-reasoning models.

## Verdicts

Each example gets one verdict and a list of issue tags:

| Verdict | Meaning |
|---|---|
| `CORRECT` | Nothing wrong |
| `MINOR_ISSUES` | Mathematically sound, but the presentation is not — broken LaTeX, or a correct `final_answer` that is not a bare value |
| `INCORRECT` | Unusable: a math error, a malformed/unsolvable question, or a `final_answer` that contradicts the solution |
| `PARSE_ERROR` | The model's reply did not contain a recognisable verdict |
| `API_ERROR` | The call failed after the SDK's 3 retries |

Issue tags: `math`, `question`, `final_answer`, `latex`. `final_answer: null` on a `PROOF` is
expected and is not flagged.

## Output

Examples are judged concurrently, so every line printed about one example carries an
`[i/N]` tag identifying it, and all of an example's output is emitted in a single write so
concurrent examples never interleave. During the run each example produces one line —
tag, verdict, category, URL — plus an indented `└─ issues: comment` line when there is
something to say. With `DEBUG=true` that is replaced by a framed block showing the
question, the full solution, the final answer and the raw model reply.

The summary printed at the end gives the run parameters, verdict counts with percentages,
issue-tag frequencies, a verdict×`category` table (which shows *which* problem types the
pipeline handles badly), and a review list — every non-`CORRECT` example grouped by
verdict, with its `[i/N]` tag, URL, issues and comment.

Per-example results are written to `data/verification/verification_results.jsonl` (one JSON
object per line): `sample_index` (the `i` from the `[i/N]` tag), the original record fields
(`source_url`, `question`, `category`, `solution`, `final_answer`), plus `verdict`,
`issues`, `comment`.
