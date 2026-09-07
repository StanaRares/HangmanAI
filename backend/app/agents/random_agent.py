from __future__ import annotations

import random

from app.agents.base import AgentDecision, LetterScore
from app.core.candidate_filter import CandidateIndex
from app.core.state import GameState


class RandomAgent:
    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def guess(self, state: GameState, index: CandidateIndex) -> AgentDecision:
        available = list(state.available_letters)
        if not available:
            raise ValueError("No letters remain to guess.")
        letter = self._rng.choice(available)
        rankings = tuple(LetterScore(letter=option, score=1.0) for option in available)
        return AgentDecision(
            agent=self.name,
            letter=letter,
            candidate_count=len(index.candidates(state)),
            rankings=rankings,
            metadata={"policy": "uniform random over unguessed letters"},
        )
