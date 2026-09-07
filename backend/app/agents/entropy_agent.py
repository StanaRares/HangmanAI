from __future__ import annotations

from app.agents.base import AgentDecision, LetterScore
from app.agents.frequency_agent import CandidateFrequencyAgent
from app.core.candidate_filter import CandidateIndex
from app.core.information import PartitionStats, partition_statistics
from app.core.state import GameState


class EntropyAgent:
    name = "entropy"

    def __init__(
        self,
        scoring: str = "risk_adjusted_entropy",
        presence_weight: float = 0.25,
        risk_weight: float = 0.7,
        name: str | None = None,
    ) -> None:
        if scoring not in {"entropy_only", "risk_adjusted_entropy"}:
            raise ValueError("Unsupported entropy scoring strategy.")
        self.name = name or ("entropy" if scoring == "entropy_only" else "risk_adjusted_entropy")
        self.scoring = scoring
        self.presence_weight = presence_weight
        self.risk_weight = risk_weight
        self._fallback = CandidateFrequencyAgent()

    def score_letter(self, state: GameState, stats: PartitionStats) -> float:
        if self.scoring == "entropy_only":
            return stats.entropy
        life_pressure = (state.max_lives + 1) / max(state.remaining_lives + 1, 1)
        miss_probability = 1.0 - stats.probability_present
        return (
            stats.entropy
            + self.presence_weight * stats.probability_present
            - self.risk_weight * miss_probability * life_pressure
        )

    def guess(self, state: GameState, index: CandidateIndex) -> AgentDecision:
        candidates = index.candidates(state)
        if not candidates:
            decision = self._fallback.guess(state, index)
            return AgentDecision(
                agent=self.name,
                letter=decision.letter,
                candidate_count=0,
                rankings=decision.rankings,
                metadata={"fallback": "candidate set empty"},
            )

        rankings = []
        for letter in state.available_letters:
            stats = partition_statistics(candidates, letter, index)
            rankings.append(
                LetterScore(
                    letter=letter,
                    score=self.score_letter(state, stats),
                    probability_present=stats.probability_present,
                    entropy=stats.entropy,
                    expected_candidates_after_guess=stats.expected_candidates_after_guess,
                    outcome_count=stats.outcome_count,
                )
            )

        rankings_tuple = tuple(
            sorted(
                rankings,
                key=lambda item: (
                    item.score,
                    item.probability_present,
                    -item.expected_candidates_after_guess,
                    item.letter,
                ),
                reverse=True,
            )
        )
        if not rankings_tuple:
            raise ValueError("No letters remain to guess.")
        return AgentDecision(
            agent=self.name,
            letter=rankings_tuple[0].letter,
            candidate_count=len(candidates),
            rankings=rankings_tuple,
            metadata={
                "scoring": self.scoring,
                "formula": (
                    "entropy + presence_weight * P(hit) - risk_weight * P(miss) "
                    "* ((max_lives + 1) / (remaining_lives + 1))"
                    if self.scoring == "risk_adjusted_entropy"
                    else "entropy"
                ),
                "presence_weight": self.presence_weight,
                "risk_weight": self.risk_weight,
            },
        )
