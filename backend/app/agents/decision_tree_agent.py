from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.agents.base import AgentDecision, LetterScore
from app.agents.frequency_agent import CandidateFrequencyAgent
from app.core.candidate_filter import CandidateIndex
from app.core.state import ALPHABET, GameState
from app.ml.features import FeatureConfig, extract_feature_dict, feature_names


class DecisionTreeAgent:
    name = "decision_tree"

    def __init__(
        self,
        model_path: str | Path | None = None,
        name: str | None = None,
    ) -> None:
        if name:
            self.name = name
        self.model_path = Path(model_path) if model_path else None
        self.model: Any | None = None
        self.feature_names = feature_names()
        self.feature_config = FeatureConfig()
        self.feature_set = "full"
        self._fallback = CandidateFrequencyAgent()

        if self.model_path and self.model_path.exists():
            self.load(self.model_path)

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def load(self, model_path: str | Path) -> None:
        import joblib

        bundle = joblib.load(model_path)
        self.model = bundle["model"] if isinstance(bundle, dict) else bundle
        if isinstance(bundle, dict):
            self.feature_names = list(bundle.get("feature_names", self.feature_names))
            config = bundle.get("feature_config", {})
            self.feature_config = FeatureConfig(**config) if config else FeatureConfig()
            self.feature_set = bundle.get("feature_set", self.feature_set)
        self.model_path = Path(model_path)

    def guess(self, state: GameState, index: CandidateIndex) -> AgentDecision:
        candidates = index.candidates(state)
        if self.model is None:
            fallback = self._fallback.guess(state, index)
            return AgentDecision(
                agent=self.name,
                letter=fallback.letter,
                candidate_count=len(candidates),
                rankings=fallback.rankings,
                metadata={
                    "model_loaded": False,
                    "fallback": "candidate_frequency",
                    "note": "Train a model with scripts/train_decision_tree.py or scripts/train_models.py.",
                },
            )

        feature_dict = extract_feature_dict(
            state,
            candidates,
            index,
            self.feature_config,
            self.feature_set,
        )
        vector = np.array([[feature_dict.get(name, 0.0) for name in self.feature_names]])
        probabilities = self._predict_probabilities(vector)
        rankings = []
        for letter in ALPHABET:
            if letter in state.guessed_letters:
                continue
            probability = probabilities.get(letter, 0.0)
            rankings.append(
                LetterScore(
                    letter=letter,
                    score=probability,
                    probability_present=feature_dict.get(f"candidate_probability_{letter}", 0.0),
                    entropy=feature_dict.get(f"entropy_{letter}", 0.0),
                    expected_candidates_after_guess=feature_dict.get(
                        f"expected_candidates_{letter}",
                        0.0,
                    ),
                    outcome_count=int(feature_dict.get(f"outcome_count_{letter}", 0.0)),
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
        if not rankings_tuple or rankings_tuple[0].score <= 0:
            fallback = self._fallback.guess(state, index)
            return AgentDecision(
                agent=self.name,
                letter=fallback.letter,
                candidate_count=len(candidates),
                rankings=fallback.rankings,
                metadata={
                    "model_loaded": True,
                    "fallback": "candidate_frequency_after_invalid_prediction",
                },
            )

        return AgentDecision(
            agent=self.name,
            letter=rankings_tuple[0].letter,
            candidate_count=len(candidates),
            rankings=rankings_tuple,
            metadata={
                "model_loaded": True,
                "model_path": str(self.model_path) if self.model_path else None,
                "top_features": self._top_features(feature_dict),
            },
        )

    def _predict_probabilities(self, vector: np.ndarray) -> dict[str, float]:
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(vector)[0]
            return {
                str(letter): float(probability)
                for letter, probability in zip(self.model.classes_, probabilities, strict=False)
            }
        prediction = str(self.model.predict(vector)[0])
        return {prediction: 1.0}

    def _top_features(self, feature_dict: dict[str, float], limit: int = 8) -> list[dict[str, float | str]]:
        if not hasattr(self.model, "feature_importances_"):
            return []
        importances = getattr(self.model, "feature_importances_")
        ranked = sorted(
            (
                (name, float(importance), float(feature_dict.get(name, 0.0)))
                for name, importance in zip(self.feature_names, importances, strict=False)
                if importance > 0
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            {"feature": name, "importance": round(importance, 6), "value": round(value, 6)}
            for name, importance, value in ranked[:limit]
        ]


class RandomForestAgent(DecisionTreeAgent):
    name = "random_forest"


class BoostingAgent(DecisionTreeAgent):
    name = "boosting"
