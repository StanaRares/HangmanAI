from __future__ import annotations

from app.core.candidate_filter import CandidateIndex, letter_pattern
from app.core.state import GameState


def state(pattern: str, guessed: set[str], wrong: set[str] | None = None) -> GameState:
    return GameState(
        word_length=len(pattern),
        pattern=pattern,
        guessed_letters=frozenset(guessed),
        incorrect_letters=frozenset(wrong or set()),
        remaining_lives=6,
        max_lives=6,
    )


def test_letter_pattern() -> None:
    assert letter_pattern("apple", "p") == "01100"
    assert letter_pattern("apple", "z") == "00000"


def test_filter_respects_known_positions_and_incorrect_letters() -> None:
    index = CandidateIndex(["caper", "cared", "baker", "later"])
    candidates = index.candidates(state("ca___", {"c", "a", "z"}, {"z"}))
    assert candidates == ("caper", "cared")


def test_filter_rejects_extra_occurrences_of_guessed_letters() -> None:
    index = CandidateIndex(["apple", "allee", "angle", "amble"])
    candidates = index.candidates(state("a___e", {"a", "e"}))
    assert candidates == ("amble", "angle", "apple")
    assert "allee" not in candidates


def test_filter_rejects_hidden_copy_of_known_letter() -> None:
    index = CandidateIndex(["cocoa", "coven", "civic"])
    candidates = index.candidates(state("co___", {"c", "o"}))
    assert candidates == ("coven",)
