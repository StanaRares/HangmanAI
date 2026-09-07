from __future__ import annotations

from app.agents.entropy_agent import EntropyAgent, partition_statistics
from app.core.candidate_filter import CandidateIndex
from app.core.state import GameState


def test_partition_statistics_entropy() -> None:
    candidates = ("cat", "car")
    stats = partition_statistics(candidates, "t")
    assert stats.entropy == 1.0
    assert stats.probability_present == 0.5
    assert stats.expected_candidates_after_guess == 1.0
    assert stats.outcome_count == 2


def test_entropy_agent_returns_rankings() -> None:
    index = CandidateIndex(["cat", "car", "cap"])
    state = GameState(
        word_length=3,
        pattern="ca_",
        guessed_letters=frozenset({"c", "a"}),
        incorrect_letters=frozenset(),
        remaining_lives=6,
        max_lives=6,
    )
    decision = EntropyAgent(scoring="entropy_only").guess(state, index)
    assert decision.letter in {"p", "r", "t"}
    assert decision.candidate_count == 3
    assert decision.rankings[0].entropy > 0
