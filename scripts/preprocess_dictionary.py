from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.dictionary import (  # noqa: E402
    PROJECT_ROOT,
    load_word_list,
    preprocess_words,
    save_words,
    split_words,
    write_split_metadata,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and split a Hangman dictionary.")
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "raw" / "sample_words.txt"),
        help="Path to a raw newline-delimited word list.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="Directory for train/validation/test word files.",
    )
    parser.add_argument("--min-length", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=15)
    parser.add_argument(
        "--allow-no-vowel",
        action="store_true",
        help="Keep words without a, e, i, o, u, or y.",
    )
    parser.add_argument(
        "--allow-roman-numerals",
        action="store_true",
        help="Keep short strings that look like Roman numerals.",
    )
    parser.add_argument(
        "--allow-long-repeats",
        action="store_true",
        help="Keep words containing four or more repeated characters in a row.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    raw_words = load_word_list(args.input)
    words = preprocess_words(
        raw_words,
        min_length=args.min_length,
        max_length=args.max_length,
        reject_no_vowel=not args.allow_no_vowel,
        reject_roman_numerals=not args.allow_roman_numerals,
        reject_long_repeats=not args.allow_long_repeats,
    )
    if len(words) < 10:
        raise SystemExit("Dictionary is too small after preprocessing.")

    train, validation, test = split_words(
        words,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    save_words(train, output_dir / "train.txt")
    save_words(validation, output_dir / "validation.txt")
    save_words(test, output_dir / "test.txt")
    save_words(words, output_dir / "vocabulary.txt")
    write_split_metadata(train, validation, test, output_dir / "dictionary_stats.json", args.seed)
    logging.info(
        "Wrote %s words: train=%s validation=%s test=%s",
        len(words),
        len(train),
        len(validation),
        len(test),
    )


if __name__ == "__main__":
    main()
