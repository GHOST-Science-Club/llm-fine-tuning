from .models import DataProcessingPipeline
from .config import INPUT_SOURCE, OUTPUT_FILE, DATASET_FILE, CHECKPOINT_FILE
import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--quiet', action='store_true', help = "Hiding console print statements while running pipeline")
    args = parser.parse_args()

    pipeline = DataProcessingPipeline(INPUT_SOURCE , OUTPUT_FILE, DATASET_FILE, CHECKPOINT_FILE, args.quiet)
    pipeline.run()

if __name__ == "__main__":
    main()
