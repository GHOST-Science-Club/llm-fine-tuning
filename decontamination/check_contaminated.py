import gzip, json
from pathlib import Path


# after running decontamination allows us to inspect the removed training examples and see which benchmark task and n-gram they matched on
for f in Path("data/decontaminated/removed").glob("*.jsonl*"):
    with gzip.open(f, "rt", encoding="utf-8") as fh:
        for line in fh:
            doc = json.loads(line)
            meta = doc.get("metadata", {})
            print("Removed training question:", doc.get("text", "")[:100])
            print("Matched benchmark task:   ", meta.get("contaminated_task"))
            print("Matched n-gram:           ", meta.get("contaminated_ngram"))
            print("---")