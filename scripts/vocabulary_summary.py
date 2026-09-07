from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.tree.vocabulary import load_vocabulary_frame, vocabulary_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the processed vocabulary summary by length.")
    parser.add_argument("--vocabulary", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = vocabulary_summary(load_vocabulary_frame(args.vocabulary))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
