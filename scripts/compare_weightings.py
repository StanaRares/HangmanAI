from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.dictionary import PROJECT_ROOT  # noqa: E402
from app.tree.evaluation import evaluate_tree  # noqa: E402
from app.tree.serialization import load_tree  # noqa: E402
from app.tree.vocabulary import load_vocabulary_frame, weights_for_length, words_for_length  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare uniform and wordfreq-weighted trees for one word length.")
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--vocabulary", default=None)
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT / "models"))
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "reports" / "results" / "weighting_comparison"),
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    frame = load_vocabulary_frame(args.vocabulary)
    words = words_for_length(frame, args.length)
    weights = weights_for_length(frame, args.length)
    rows = []
    for tree_weighting in ("uniform", "wordfreq"):
        model_path = Path(args.models_dir) / f"length_{args.length}" / tree_weighting / "tree.json.gz"
        if not model_path.exists():
            logging.info("Skipping missing %s tree at %s", tree_weighting, model_path)
            continue
        tree = load_tree(model_path)
        summary, _ = evaluate_tree(tree, words, model_path, weights=weights)
        rows.append(
            {
                "length": args.length,
                "tree_weighting": tree_weighting,
                "best_first_guess": summary["best_first_guess"],
                "unweighted_win_rate": summary["win_rate"],
                "wordfreq_weighted_win_rate": summary["weighted_win_rate"],
                "unweighted_average_mistakes": summary["average_wrong_guesses"],
                "wordfreq_weighted_average_mistakes": summary["weighted_average_wrong_guesses"],
                "node_count": summary["node_count"],
                "maximum_tree_depth": summary["maximum_tree_depth"],
                "training_strategy": summary["training_strategy"],
                "training_time": summary["training_time"],
            }
        )
    output_dir = Path(args.output_dir)
    write_csv(output_dir / f"length_{args.length}.csv", rows)
    (output_dir / f"length_{args.length}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    logging.info("Wrote weighting comparison for length %s", args.length)


if __name__ == "__main__":
    main()
