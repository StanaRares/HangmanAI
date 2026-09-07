from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from app.core.candidate_filter import CandidateIndex
from app.core.state import GameState


@dataclass(frozen=True)
class LetterScore:
    letter: str
    score: float
    probability_present: float = 0.0
    entropy: float = 0.0
    expected_candidates_after_guess: float = 0.0
    outcome_count: int = 0

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "letter": self.letter,
            "score": round(self.score, 6),
            "probability_present": round(self.probability_present, 6),
            "entropy": round(self.entropy, 6),
            "expected_candidates_after_guess": round(
                self.expected_candidates_after_guess,
                6,
            ),
            "outcome_count": self.outcome_count,
        }


@dataclass(frozen=True)
class AgentDecision:
    agent: str
    letter: str
    candidate_count: int
    rankings: tuple[LetterScore, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "guess": self.letter,
            "candidate_count": self.candidate_count,
            "rankings": [score.to_dict() for score in self.rankings],
            "metadata": self.metadata,
        }


class BaseAgent(Protocol):
    name: str

    def guess(self, state: GameState, index: CandidateIndex) -> AgentDecision:
        ...
