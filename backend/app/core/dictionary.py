from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable

WORD_RE = re.compile(r"^[a-z]+$")
VOWEL_RE = re.compile(r"[aeiouy]")
ROMAN_RE = re.compile(r"^[ivxlcdm]+$")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def clean_word(word: str) -> str:
    """Normalize a dictionary entry and reject non alphabetic tokens."""
    normalized = word.strip().lower()
    if not normalized or not WORD_RE.fullmatch(normalized):
        return ""
    return normalized


def preprocess_words(
    words: Iterable[str],
    min_length: int = 4,
    max_length: int = 15,
    reject_no_vowel: bool = True,
    reject_roman_numerals: bool = True,
    reject_long_repeats: bool = True,
) -> list[str]:
    cleaned = set()
    for raw in words:
        word = clean_word(raw)
        if not word or not min_length <= len(word) <= max_length:
            continue
        if reject_no_vowel and not VOWEL_RE.search(word):
            continue
        if reject_roman_numerals and len(word) <= 6 and ROMAN_RE.fullmatch(word):
            continue
        if reject_long_repeats and re.search(r"(.)\1{3,}", word):
            continue
        cleaned.add(word)
    return sorted(cleaned)


def load_word_list(path: str | Path) -> list[str]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def read_words(path: str | Path) -> list[str]:
    return preprocess_words(load_word_list(path), min_length=1, max_length=64)


def save_words(words: Iterable[str], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for word in sorted(set(words)):
            handle.write(f"{word}\n")


def split_words(
    words: list[str],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1.")

    shuffled = list(words)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    validation_end = train_end + int(len(shuffled) * validation_ratio)
    return (
        sorted(shuffled[:train_end]),
        sorted(shuffled[train_end:validation_end]),
        sorted(shuffled[validation_end:]),
    )


def write_split_metadata(
    train: list[str],
    validation: list[str],
    test: list[str],
    path: str | Path,
    seed: int,
) -> None:
    metadata = {
        "seed": seed,
        "train_count": len(train),
        "validation_count": len(validation),
        "test_count": len(test),
        "total_count": len(train) + len(validation) + len(test),
        "lengths": sorted({len(word) for word in train + validation + test}),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def default_dictionary_paths() -> list[Path]:
    return [
        PROJECT_ROOT / "data" / "processed" / "train.txt",
        PROJECT_ROOT / "data" / "processed" / "validation.txt",
        PROJECT_ROOT / "data" / "processed" / "test.txt",
        PROJECT_ROOT / "data" / "raw" / "sample_words.txt",
    ]


def load_default_words() -> list[str]:
    combined: list[str] = []
    processed_paths = default_dictionary_paths()[:3]
    if all(path.exists() for path in processed_paths):
        for path in processed_paths:
            combined.extend(read_words(path))
        return sorted(set(combined))

    sample = default_dictionary_paths()[-1]
    if sample.exists():
        return read_words(sample)
    raise FileNotFoundError("No dictionary found. Add data/raw/sample_words.txt or processed splits.")
