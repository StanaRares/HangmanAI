from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.dictionary import PROJECT_ROOT  # noqa: E402
from app.ml.features import FeatureConfig, feature_names  # noqa: E402


def main() -> None:
    config = FeatureConfig()
    public_state = set(feature_names(config, "public_state"))
    candidate_stats = set(feature_names(config, "candidate_stats"))
    full = set(feature_names(config, "full"))
    entropy_derived = sorted(name for name in full if name.startswith(("entropy_", "expected_candidates_", "outcome_count_")))
    candidate_only = sorted(candidate_stats - public_state)
    report = {
        "summary": (
            "No feature contains the hidden word or any future outcome. Candidate statistics are "
            "computed from the public pattern and guessed letters. Entropy-derived features are "
            "available during gameplay but can cause the model to imitate the entropy teacher, so "
            "they are separated into the full feature-set ablation."
        ),
        "feature_sets": {
            "public_state": {
                "count": len(public_state),
                "description": "Only word length, lives, turn, revealed pattern, guessed/incorrect/known letter vectors.",
            },
            "candidate_stats": {
                "count": len(candidate_stats),
                "description": "Public state plus candidate count and candidate letter document probabilities.",
                "added_features": candidate_only,
            },
            "full": {
                "count": len(full),
                "description": "Candidate stats plus entropy-derived split features.",
                "entropy_derived_feature_count": len(entropy_derived),
            },
        },
        "leakage_checks": [
            "Agents and feature extraction receive GameState only, never the environment solution.",
            "Training rows may store hidden_word for provenance, but training scripts exclude it from feature_names.",
            "model_generalization refuses overlapping train/test word splits.",
            "train_models.py refuses training data whose hidden_word metadata overlaps configured test words.",
        ],
        "redundancy_notes": [
            "candidate_count and log_candidate_count are intentionally redundant for tree interpretability.",
            "guessed_count is derivable from guessed_* but useful for shallow trees.",
            "known_* is derivable from pattern positions but improves readability of feature importances.",
        ],
    }
    output = PROJECT_ROOT / "reports" / "results" / "feature_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
