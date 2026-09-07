from __future__ import annotations

import importlib.metadata
import json
import platform
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.dictionary import PROJECT_ROOT

ASCII_WORD_RE = re.compile(r"^[a-z]+$")
VOWEL_RE = re.compile(r"[aeiouy]")


def clean_hangman_word(token: str, min_length: int = 3) -> str:
    word = token.strip().lower()
    if len(word) < min_length:
        return ""
    if not ASCII_WORD_RE.fullmatch(word):
        return ""
    if re.search(r"(.)\1{4,}", word):
        return ""
    if len(word) > 3 and not VOWEL_RE.search(word):
        return ""
    return word


def build_wordfreq_vocabulary(min_length: int = 3) -> pd.DataFrame:
    from wordfreq import iter_wordlist, zipf_frequency

    words: dict[str, float] = {}
    for token in iter_wordlist("en", wordlist="best"):
        word = clean_hangman_word(token, min_length=min_length)
        if word and word not in words:
            words[word] = float(zipf_frequency(word, "en", wordlist="best"))
    frame = pd.DataFrame(
        {
            "word": list(words.keys()),
            "length": [len(word) for word in words],
            "zipf_frequency": list(words.values()),
        }
    )
    return frame.sort_values(["length", "word"]).reset_index(drop=True)


def vocabulary_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("length", as_index=False)
        .agg(
            number_of_words=("word", "count"),
            mean_zipf_frequency=("zipf_frequency", "mean"),
            max_zipf_frequency=("zipf_frequency", "max"),
            min_zipf_frequency=("zipf_frequency", "min"),
        )
        .sort_values("length")
    )


def write_vocabulary_outputs(
    frame: pd.DataFrame,
    output_dir: str | Path | None = None,
    min_length: int = 3,
) -> dict[str, Any]:
    root = Path(output_dir) if output_dir else PROJECT_ROOT / "data" / "processed"
    by_length = root / "by_length"
    by_length.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(root / "wordfreq_vocabulary.parquet", index=False)
    frame[["word"]].to_csv(root / "wordfreq_words.txt", index=False, header=False)

    for length, group in frame.groupby("length"):
        words = group["word"].sort_values()
        (by_length / f"words_{int(length)}.txt").write_text(
            "\n".join(words) + "\n",
            encoding="utf-8",
        )

    summary = vocabulary_summary(frame)
    summary.to_csv(root / "wordfreq_vocabulary_summary.csv", index=False)
    metadata = {
        "source": "wordfreq.iter_wordlist('en', wordlist='best')",
        "wordfreq_version": importlib.metadata.version("wordfreq"),
        "python_version": platform.python_version(),
        "vocabulary_extraction_date": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "min_length": min_length,
            "ascii_lowercase_a_to_z_only": True,
            "requires_vowel_for_length_gt_3": True,
            "rejects_five_or_more_repeated_characters": True,
        },
        "total_usable_words": int(len(frame)),
        "lengths": {
            str(int(row.length)): int(row.number_of_words)
            for row in summary.itertuples(index=False)
        },
    }
    (root / "wordfreq_vocabulary_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def load_vocabulary_frame(path: str | Path | None = None) -> pd.DataFrame:
    source = Path(path) if path else PROJECT_ROOT / "data" / "processed" / "wordfreq_vocabulary.parquet"
    if source.exists():
        return pd.read_parquet(source)
    fallback = PROJECT_ROOT / "data" / "processed" / "vocabulary.txt"
    if fallback.exists():
        words = [line.strip() for line in fallback.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        words = [line.strip() for line in (PROJECT_ROOT / "data" / "raw" / "sample_words.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(
        {
            "word": words,
            "length": [len(word) for word in words],
            "zipf_frequency": [1.0 for _ in words],
        }
    )


def words_for_length(frame: pd.DataFrame, length: int) -> list[str]:
    return sorted(frame.loc[frame["length"] == length, "word"].astype(str).tolist())


def weights_for_length(frame: pd.DataFrame, length: int) -> dict[str, float]:
    subset = frame.loc[frame["length"] == length, ["word", "zipf_frequency"]]
    weights = {}
    for word, zipf in subset.itertuples(index=False):
        # zipf is log10 occurrences per billion. Use positive linear weights
        # but cap the exponent enough to avoid floating-point domination.
        weights[str(word)] = 10 ** max(min(float(zipf), 8.0), 0.0)
    return weights
