from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

from app.core.candidate_filter import CandidateIndex
from app.core.information import partition_statistics
from app.core.state import ALPHABET, GameState, UNKNOWN


@dataclass(frozen=True)
class FeatureConfig:
    max_word_length: int = 20


FeatureSet = Literal["public_state", "candidate_stats", "full"]


def _letter_code(char: str) -> int:
    if char == UNKNOWN:
        return 0
    return ALPHABET.index(char) + 1


def feature_names(
    config: FeatureConfig | None = None,
    feature_set: FeatureSet = "full",
) -> list[str]:
    """Return deterministic feature names for the requested ablation set.

    public_state:
        Uses only information visible in the public Hangman state.
    candidate_stats:
        Adds candidate count and candidate letter probabilities.
    full:
        Adds entropy-derived features. This is useful as an upper-feature
        baseline, but it can make the tree mimic the entropy teacher.
    """

    cfg = config or FeatureConfig()
    names = [
        "word_length",
        "remaining_lives",
        "max_lives",
        "turn",
        "revealed_count",
        "revealed_fraction",
        "guessed_count",
        "incorrect_count",
    ]
    names.extend(f"pattern_pos_{index}" for index in range(cfg.max_word_length))
    for prefix in ("guessed", "incorrect", "known"):
        names.extend(f"{prefix}_{letter}" for letter in ALPHABET)

    if feature_set in {"candidate_stats", "full"}:
        names.extend(["candidate_count", "log_candidate_count"])
        names.extend(f"candidate_probability_{letter}" for letter in ALPHABET)

    if feature_set == "full":
        for prefix in ("entropy", "expected_candidates", "outcome_count"):
            names.extend(f"{prefix}_{letter}" for letter in ALPHABET)

    return names


def all_feature_names(config: FeatureConfig | None = None) -> list[str]:
    return feature_names(config, "full")


def extract_feature_dict(
    state: GameState,
    candidates: tuple[str, ...],
    index: CandidateIndex | None = None,
    config: FeatureConfig | None = None,
    feature_set: FeatureSet = "full",
) -> OrderedDict[str, float]:
    cfg = config or FeatureConfig()
    total_candidates = len(candidates)
    features: OrderedDict[str, float] = OrderedDict()
    features["word_length"] = float(state.word_length)
    features["remaining_lives"] = float(state.remaining_lives)
    features["max_lives"] = float(state.max_lives)
    features["turn"] = float(state.turn)
    features["revealed_count"] = float(state.revealed_count)
    features["revealed_fraction"] = (
        state.revealed_count / state.word_length if state.word_length else 0.0
    )
    features["guessed_count"] = float(len(state.guessed_letters))
    features["incorrect_count"] = float(len(state.incorrect_letters))

    for position in range(cfg.max_word_length):
        if position < len(state.pattern):
            features[f"pattern_pos_{position}"] = float(_letter_code(state.pattern[position]))
        else:
            features[f"pattern_pos_{position}"] = -1.0

    known_letters = state.correct_letters

    for letter in ALPHABET:
        features[f"guessed_{letter}"] = float(letter in state.guessed_letters)
    for letter in ALPHABET:
        features[f"incorrect_{letter}"] = float(letter in state.incorrect_letters)
    for letter in ALPHABET:
        features[f"known_{letter}"] = float(letter in known_letters)

    if feature_set in {"candidate_stats", "full"}:
        probabilities = (
            index.letter_probabilities(candidates)
            if index is not None
            else _unweighted_letter_probabilities(candidates)
        )
        features["candidate_count"] = float(total_candidates)
        features["log_candidate_count"] = math.log1p(total_candidates)
        for letter in ALPHABET:
            features[f"candidate_probability_{letter}"] = float(probabilities[letter])

    if feature_set == "full":
        stats_by_letter = {
            letter: partition_statistics(candidates, letter, index) for letter in ALPHABET
        }
        for letter in ALPHABET:
            stats = stats_by_letter[letter]
            features[f"entropy_{letter}"] = float(stats.entropy)
        for letter in ALPHABET:
            stats = stats_by_letter[letter]
            features[f"expected_candidates_{letter}"] = float(
                stats.expected_candidates_after_guess
            )
        for letter in ALPHABET:
            stats = stats_by_letter[letter]
            features[f"outcome_count_{letter}"] = float(stats.outcome_count)

    return features


def extract_features(
    state: GameState,
    candidates: tuple[str, ...],
    index: CandidateIndex | None = None,
    config: FeatureConfig | None = None,
    feature_set: FeatureSet = "full",
) -> list[float]:
    features = extract_feature_dict(state, candidates, index, config, feature_set)
    return [features[name] for name in feature_names(config, feature_set)]


def _unweighted_letter_probabilities(candidates: tuple[str, ...]) -> dict[str, float]:
    if not candidates:
        return {letter: 0.0 for letter in ALPHABET}
    return {
        letter: sum(1 for word in candidates if letter in word) / len(candidates)
        for letter in ALPHABET
    }
