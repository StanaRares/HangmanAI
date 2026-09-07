from __future__ import annotations

from app.core.candidate_filter import CandidateIndex
from app.core.state import GameState
from app.ml.features import FeatureConfig, extract_feature_dict, extract_features, feature_names


def test_feature_order_is_deterministic() -> None:
    config = FeatureConfig(max_word_length=6)
    names = feature_names(config)
    assert names == feature_names(config)
    assert names[0:4] == ["word_length", "remaining_lives", "max_lives", "turn"]
    assert "candidate_probability_a" in names
    assert "outcome_count_z" in names


def test_extract_features_matches_names() -> None:
    index = CandidateIndex(["apple", "angle", "amble"])
    state = GameState(
        word_length=5,
        pattern="a___e",
        guessed_letters=frozenset({"a", "e"}),
        incorrect_letters=frozenset(),
        remaining_lives=5,
        max_lives=6,
    )
    candidates = index.candidates(state)
    vector = extract_features(state, candidates, index)
    features = extract_feature_dict(state, candidates, index)
    assert len(vector) == len(feature_names())
    assert list(features) == feature_names()
    assert features["candidate_count"] == 3.0
    assert features["known_a"] == 1.0
    assert features["guessed_e"] == 1.0
