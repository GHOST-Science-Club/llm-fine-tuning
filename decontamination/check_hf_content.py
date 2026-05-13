


from datasets import load_from_disk

#ds = load_from_disk("data/decontaminated/hf_dataset")
ds = load_from_disk("data/dataset/processed")

for row in ds:
    for col, val in row.items():
        print(f"{col}::: {val}")
    print("---")