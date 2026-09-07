from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import plot_tree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.dictionary import PROJECT_ROOT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Decision Tree visual reports.")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models" / "decision_tree.joblib"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports" / "figures"))
    parser.add_argument("--max-depth", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model)
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    features = bundle.get("feature_names", []) if isinstance(bundle, dict) else []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    importances = pd.DataFrame(
        {
            "feature": features,
            "importance": getattr(model, "feature_importances_", []),
        }
    ).sort_values("importance", ascending=False)
    importances.to_csv(output_dir / "feature_importances.csv", index=False)

    top = importances.head(20).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["importance"], color="#54d6a7")
    plt.title("Top Decision Tree Feature Importances")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importances.png", dpi=160)
    plt.close()

    plt.figure(figsize=(18, 10))
    plot_tree(
        model,
        max_depth=args.max_depth,
        feature_names=features,
        class_names=[str(item) for item in getattr(model, "classes_", [])],
        filled=True,
        rounded=True,
        fontsize=7,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "decision_tree_first_levels.png", dpi=180)
    plt.close()

    summary = {
        "tree_depth": model.get_depth(),
        "node_count": model.tree_.node_count,
        "top_features": importances.head(20).to_dict(orient="records"),
    }
    (output_dir / "decision_tree_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    logging.info("Wrote visual reports to %s", output_dir)


if __name__ == "__main__":
    main()
