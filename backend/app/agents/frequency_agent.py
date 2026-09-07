from __future__ import annotations

from collections import Counter

from app.agents.base import AgentDecision, LetterScore
from app.core.candidate_filter import CandidateIndex
from app.core.state import ALPHABET, GameState


class GlobalFrequencyAgent:
    name = "global_frequency"

    def __init__(self, words: tuple[str, ...] | list[str]) -> None:
        counts: Counter[str] = Counter()
        for word in words:
            counts.update(set(word))
        total = max(len(words), 1)
        self.probabilities = {letter: counts[letter] / total for letter in ALPHABET}

    def guess(self, state: GameState, index: CandidateIndex) -> AgentDecision:
        rankings = tuple(
            sorted(
                (
                    LetterScore(
                        letter=letter,
                        score=self.probabilities[letter],
                        probability_present=self.probabilities[letter],
                    )
                    for letter in state.available_letters
                ),
                key=lambda item: (item.score, item.letter),
                reverse=True,
            )
        )
        if not rankings:
            raise ValueError("No letters remain to guess.")
        return AgentDecision(
            agent=self.name,
            letter=rankings[0].letter,
            candidate_count=len(index.candidates(state)),
            rankings=rankings,
            metadata={"frequency_definition": "document frequency across the training dictionary"},
        )


class CandidateFrequencyAgent:
    name = "candidate_frequency"

    def guess(self, state: GameState, index: CandidateIndex) -> AgentDecision:
        candidates = index.candidates(state)
        probabilities = index.letter_probabilities(candidates)
        rankings = tuple(
            sorted(
                (
                    LetterScore(
                        letter=letter,
                        score=probabilities[letter],
                        probability_present=probabilities[letter],
                    )
                    for letter in state.available_letters
                ),
                key=lambda item: (item.score, item.letter),
                reverse=True,
            )
        )
        if not rankings:
            raise ValueError("No letters remain to guess.")
        return AgentDecision(
            agent=self.name,
            letter=rankings[0].letter,
            candidate_count=len(candidates),
            rankings=rankings,
            metadata={
                "frequency_definition": (
                    "document frequency within compatible candidates; repeated letters "
                    "inside one candidate count once"
                )
            },
        )
