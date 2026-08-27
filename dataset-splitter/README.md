# Dataset Splitter Module

A self-contained, modular Python package for splitting JSONL datasets into subsets for Supervised Fine-Tuning (**SFT**) and Reinforcement Learning (**GRPO**).

Designed to be dropped directly into any repository, imported as a Python package, or run standalone via CLI.

---

## Module Structure

```text
dataset-splitter/               # Module root folder 
├── __init__.py                 # Package initializer
├── __main__.py                 # CLI entry point for module execution
├── config.py                   # Config dataclass, absolute path resolution, CLI parser
├── splitter.py                 # Dataset loading, shuffling, and splitting logic
├── .gitignore                  # Git ignore rules for datasets, venv, and cache
├── data/
│   ├── raw/                    # Raw / unformatted source data before preprocessing
│   └── formatted/              # Formatted dataset (e.g. deduplicated.jsonl) & split outputs
└── README.md                   # Documentation
```

### Data Directories
- **`data/raw/`**: Place raw, unformatted datasets here (before applying cleaning, deduplication, or prompt/response formatting).
- **`data/formatted/`**: Stores cleaned/formatted JSONL datasets (such as `deduplicated.jsonl`) and receives the split outputs (`sft.jsonl` and `grpo.jsonl`).

---

## CLI Usage

### Module execution
Run from the root project directory (`llm-fine-tuning`):

```bash
python -m dataset-splitter
```

### Custom Arguments
```bash
python -m dataset-splitter \
  --input dataset-splitter/data/formatted/deduplicated.jsonl \
  --sft_out dataset-splitter/data/formatted/sft.jsonl \
  --grpo_out dataset-splitter/data/formatted/grpo.jsonl \
  --sft_ratio 0.2 \
  --seed 123
```

#### Available Options

| Argument | Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | `<module_root>/data/formatted/deduplicated.jsonl` | Absolute path to the source JSONL dataset file |
| `--sft_out` | | `<module_root>/data/formatted/sft.jsonl` | Absolute path to the output SFT dataset file |
| `--grpo_out` | | `<module_root>/data/formatted/grpo.jsonl` | Absolute path to the output GRPO dataset file |
| `--sft_ratio` | | `0.1` | Proportion of dataset allocated to SFT (e.g. `0.1` = 10%) |
| `--seed` | | `42` | Random seed for deterministic and reproducible shuffling |
| `--help` | `-h` | | Show help message and exit |

---


