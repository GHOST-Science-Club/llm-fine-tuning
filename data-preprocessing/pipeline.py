from .models import DataProcessingPipeline
from .config import config
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--quiet', action='store_true', help="Hiding console print statements while running pipeline")
    args = parser.parse_args()

    pipeline = DataProcessingPipeline(config.INPUT_SOURCE, config.OUTPUT_FILE, config.DATASET_FILE, config.CHECKPOINT_FILE, args.quiet)
    pipeline.run()

if __name__ == "__main__":
    main()
