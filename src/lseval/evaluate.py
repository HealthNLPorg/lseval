import argparse
import logging

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--corpus_json",
    required=True,
    help="Exported Label Studio JSON.",
)
parser.add_argument(
    "--per_document",
    action="store_true",
    help="Print out scores for each document, rather than overall scores",
)
parser.add_argument(
    "--overlap",
    dest="spans_type",
    action="store_true",
    help="Count predicted annotation spans as correct if they overlap by one character or more "
    + "with a reference annotation span. Not intended as a real evaluation method (since what "
    + "to do with multiple matches is not well defined) but useful for debugging purposes.",
)
logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def score(corpus_json: str, per_document: bool, overlap: bool) -> None:
    pass


def main() -> None:
    args = parser.parse_args()
    score(
        corpus_json=args.corpus_json,
        per_document=args.per_document,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
