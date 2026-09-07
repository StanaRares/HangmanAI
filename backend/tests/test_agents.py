from __future__ import annotations

from app.agents import CandidateFrequencyAgent, GlobalFrequencyAgent, RandomAgent
from app.core.candidate_filter import CandidateIndex
from app.core.state import GameState


def sample_state() -> GameState:
    return GameState(
        word_length=5,
        pattern="a___e",
        guessed_letters=frozenset({"a", "e"}),
        incorrect_letters=frozenset(),
        remaining_lives=6,
        max_lives=6,
    )


def test_candidate_frequency_uses_document_frequency() -> None:
    index = CandidateIndex(["apple", "angle", "amble"])
    decision = CandidateFrequencyAgent().guess(sample_state(), index)
    assert decision.letter == "l"
    assert decision.rankings[0].probability_present == 1.0


def test_global_frequency_ignores_guessed_letters() -> None:
    index = CandidateIndex(["aaa", "bbb", "ccc"])
    state = GameState(
        word_length=3,
        pattern="___",
        guessed_letters=frozenset({"a"}),
        incorrect_letters=frozenset(),
        remaining_lives=6,
        max_lives=6,
    )
    decision = GlobalFrequencyAgent(["aaa", "bbb", "ccc"]).guess(state, index)
    assert decision.letter != "a"


def test_random_agent_only_chooses_available_letters() -> None:
    index = CandidateIndex(["abc"])
    state = GameState(
        word_length=3,
        pattern="a__",
        guessed_letters=frozenset({"a"}),
        incorrect_letters=frozenset(),
        remaining_lives=6,
        max_lives=6,
    )
    decision = RandomAgent(seed=1).guess(state, index)
    assert decision.letter != "a"
