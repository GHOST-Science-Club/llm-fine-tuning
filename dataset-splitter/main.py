from config import parse_args
from splitter import split_dataset


def main() -> None:
    config = parse_args()
    config.ensure_directories()
    split_dataset(config)


if __name__ == "__main__":
    main()
