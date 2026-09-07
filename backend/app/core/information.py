from __future__ import annotations

import math
from dataclasses import dataclass

from app.core.candidate_filter import CandidateIndex, letter_pattern


@dataclass(frozen=True)
class PartitionStats:
    entropy: float
    probability_present: float
    expected_candidates_after_guess: float
    outcome_count: int


def partition_statistics(
    candidates: tuple[str, ...],
    letter: str,
    index: CandidateIndex | None = None,
) -> PartitionStats:
    if not candidates:
        return PartitionStats(
            entropy=0.0,
            probability_present=0.0,
            expected_candidates_after_guess=0.0,
            outcome_count=0,
        )

    weights_by_pattern: dict[str, float] = {}
    counts_by_pattern: dict[str, int] = {}
    total_weight = 0.0
    present_weight = 0.0
    absent_pattern = "0" * len(candidates[0])

    for word in candidates:
        weight = index.weight(word) if index else 1.0
        pattern = letter_pattern(word, letter)
        weights_by_pattern[pattern] = weights_by_pattern.get(pattern, 0.0) + weight
        counts_by_pattern[pattern] = counts_by_pattern.get(pattern, 0) + 1
        total_weight += weight
        if pattern != absent_pattern:
            present_weight += weight

    if total_weight <= 0:
        return PartitionStats(
            entropy=0.0,
            probability_present=0.0,
            expected_candidates_after_guess=0.0,
            outcome_count=len(weights_by_pattern),
        )

    entropy = 0.0
    expected_candidates = 0.0
    for pattern, weight in weights_by_pattern.items():
        probability = weight / total_weight
        entropy -= probability * math.log2(probability)
        expected_candidates += probability * counts_by_pattern[pattern]

    return PartitionStats(
        entropy=entropy,
        probability_present=present_weight / total_weight,
        expected_candidates_after_guess=expected_candidates,
        outcome_count=len(weights_by_pattern),
    )
