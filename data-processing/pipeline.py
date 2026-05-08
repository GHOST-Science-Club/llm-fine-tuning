from .models import DataProcessingPipeline
from .config import INPUT_FILE, OUTPUT_FILE, DATASET_FILE, CHECKPOINT_FILE

def main():
    pipeline = DataProcessingPipeline(INPUT_FILE, OUTPUT_FILE, DATASET_FILE, CHECKPOINT_FILE)
    pipeline.run()

if __name__ == "__main__":
    main()
