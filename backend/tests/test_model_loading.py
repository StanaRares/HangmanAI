from __future__ import annotations

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from app.agents.decision_tree_agent import DecisionTreeAgent
from app.core.candidate_filter import CandidateIndex
from app.core.state import GameState
from app.ml.features import FeatureConfig, feature_names


def test_decision_tree_agent_loads_model(tmp_path) -> None:
    config = FeatureConfig()
    names = feature_names(config)
    x = np.zeros((4, len(names)))
    x[0, 0] = 5
    x[1, 0] = 6
    x[2, 0] = 5
    x[3, 0] = 6
    y = np.array(["a", "b", "a", "b"])
    model = DecisionTreeClassifier(random_state=42).fit(x, y)
    path = tmp_path / "tree.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_names": names,
            "feature_config": {"max_word_length": config.max_word_length},
        },
        path,
    )

    agent = DecisionTreeAgent(path)
    index = CandidateIndex(["apple", "angle"])
    state = GameState(
        word_length=5,
        pattern="_____",
        guessed_letters=frozenset(),
        incorrect_letters=frozenset(),
        remaining_lives=6,
        max_lives=6,
    )
    decision = agent.guess(state, index)
    assert agent.is_trained
    assert decision.letter in {"a", "b"}
    assert decision.metadata["model_loaded"] is True
