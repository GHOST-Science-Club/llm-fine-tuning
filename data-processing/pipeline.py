from pathlib import Path
from models import DataProcessingPipeline

def main():
    pipeline = DataProcessingPipeline(Path("./data/input/forum_example_fixed.jsonl"), Path("./data/output"), Path("./data/dataset/processed"))
    pipeline.run()

if __name__ == "__main__":
    main()
