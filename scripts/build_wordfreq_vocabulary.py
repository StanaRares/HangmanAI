from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.dictionary import PROJECT_ROOT  # noqa: E402
from app.tree.vocabulary import build_wordfreq_vocabulary, write_vocabulary_outputs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract every usable English entry exposed by wordfreq.iter_wordlist."
    )
    parser.add_argument("--min-length", type=int, default=3)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = build_wordfreq_vocabulary(min_length=args.min_length)
    metadata = write_vocabulary_outputs(frame, args.output_dir, min_length=args.min_length)
    logging.info("Extracted %s usable words from %s", metadata["total_usable_words"], metadata["source"])
    for length, count in metadata["lengths"].items():
        logging.info("length %s: %s words", length, count)


if __name__ == "__main__":
    main()
