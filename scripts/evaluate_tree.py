from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.dictionary import PROJECT_ROOT  # noqa: E402
from app.tree.evaluation import evaluate_tree  # noqa: E402
from app.tree.serialization import load_tree  # noqa: E402
from app.tree.vocabulary import load_vocabulary_frame, weights_for_length, words_for_length  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one saved Hangman decision tree.")
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--weighting", choices=["uniform", "wordfreq"], default="uniform")
    parser.add_argument("--vocabulary", default=None)
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports" / "results" / "trees"))
    parser.add_argument("--max-words", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.models_dir) / f"length_{args.length}" / args.weighting / "tree.json.gz"
    tree = load_tree(model_path)
    frame = load_vocabulary_frame(args.vocabulary)
    words = words_for_length(frame, args.length)
    weights = weights_for_length(frame, args.length) if args.weighting == "wordfreq" else None
    if args.max_words:
        words = words[: args.max_words]
        if weights:
            weights = {word: weights[word] for word in words if word in weights}
    summary, games = evaluate_tree(tree, words, model_path, weights=weights)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"length_{args.length}_{args.weighting}_summary.json"
    games_path = output_dir / f"length_{args.length}_{args.weighting}_games.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with games_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(games[0]))
        writer.writeheader()
        writer.writerows(games)
    logging.info("Wrote %s and %s", summary_path, games_path)


if __name__ == "__main__":
    main()
