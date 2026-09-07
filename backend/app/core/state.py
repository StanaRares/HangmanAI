from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ALPHABET: tuple[str, ...] = tuple("abcdefghijklmnopqrstuvwxyz")
UNKNOWN = "_"
GameStatus = Literal["playing", "won", "lost"]


def normalize_letter(letter: str) -> str:
    """Return a normalized single ASCII letter or raise ValueError."""
    normalized = letter.strip().lower()
    if len(normalized) != 1 or normalized not in ALPHABET:
        raise ValueError("Guess must be a single letter a-z.")
    return normalized


@dataclass(frozen=True)
class GuessRecord:
    letter: str
    correct: bool
    positions: tuple[int, ...]
    pattern: str
    remaining_lives: int

    def to_dict(self) -> dict[str, object]:
        return {
            "letter": self.letter,
            "correct": self.correct,
            "positions": list(self.positions),
            "pattern": self.pattern,
            "remaining_lives": self.remaining_lives,
        }


@dataclass(frozen=True)
class GameState:
    """Public state exposed to agents. It intentionally never includes the word."""

    word_length: int
    pattern: str
    guessed_letters: frozenset[str]
    incorrect_letters: frozenset[str]
    remaining_lives: int
    max_lives: int
    status: GameStatus = "playing"
    turn: int = 0
    history: tuple[GuessRecord, ...] = field(default_factory=tuple)

    @property
    def correct_letters(self) -> frozenset[str]:
        return frozenset(letter for letter in self.pattern if letter != UNKNOWN)

    @property
    def revealed_count(self) -> int:
        return sum(1 for char in self.pattern if char != UNKNOWN)

    @property
    def available_letters(self) -> tuple[str, ...]:
        return tuple(letter for letter in ALPHABET if letter not in self.guessed_letters)

    @property
    def is_finished(self) -> bool:
        return self.status in {"won", "lost"}

    def to_public_dict(self) -> dict[str, object]:
        return {
            "word_length": self.word_length,
            "pattern": self.pattern,
            "pattern_display": " ".join(self.pattern),
            "guessed_letters": sorted(self.guessed_letters),
            "correct_letters": sorted(self.correct_letters),
            "incorrect_letters": sorted(self.incorrect_letters),
            "remaining_lives": self.remaining_lives,
            "max_lives": self.max_lives,
            "status": self.status,
            "turn": self.turn,
            "history": [record.to_dict() for record in self.history],
        }
