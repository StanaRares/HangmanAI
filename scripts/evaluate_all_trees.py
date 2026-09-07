from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.dictionary import PROJECT_ROOT  # noqa: E402
from app.tree.evaluation import evaluate_tree  # noqa: E402
from app.tree.serialization import load_tree  # noqa: E402
from app.tree.vocabulary import load_vocabulary_frame, weights_for_length, words_for_length  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate every trained word-length tree.")
    parser.add_argument("--weighting", choices=["uniform", "wordfreq"], default="uniform")
    parser.add_argument("--vocabulary", default=None)
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports" / "results"))
    parser.add_argument("--max-words", type=int, default=None)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows: list[dict[str, Any]], metric: str, output: Path, ylabel: str) -> None:
    if not rows:
        return
    lengths = [int(row["length"]) for row in rows]
    values = [float(row[metric]) for row in rows]
    plt.figure(figsize=(8, 4.5))
    plt.plot(lengths, values, marker="o", color="#54d6a7")
    plt.xlabel("Word length")
    plt.ylabel(ylabel)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close()


def hardest_rows(length: int, games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    max_total = max(int(game["total_guesses"]) for game in games) if games else 0
    for row in games:
        category = ""
        if row["won"] is False:
            category = "failed"
        elif int(row["incorrect_guesses"]) == 5:
            category = "solved_with_five_mistakes"
        elif int(row["incorrect_guesses"]) == 0:
            category = "solved_with_no_mistakes"
        if int(row["total_guesses"]) == max_total:
            category = "most_total_guesses" if not category else f"{category};most_total_guesses"
        if category:
            output.append({"length": length, "category": category, **row})
    return output


def main() -> None:
    args = parse_args()
    frame = load_vocabulary_frame(args.vocabulary)
    models_dir = Path(args.models_dir)
    summary_rows = []
    hard_rows = []
    for model_path in sorted(models_dir.glob(f"length_*/{args.weighting}/tree.json.gz")):
        length = int(model_path.parents[1].name.split("_")[1])
        tree = load_tree(model_path)
        words = words_for_length(frame, length)
        weights = weights_for_length(frame, length) if args.weighting == "wordfreq" else None
        if args.max_words:
            words = words[: args.max_words]
            if weights:
                weights = {word: weights[word] for word in words if word in weights}
        logging.info("Evaluating length %s on %s words", length, len(words))
        summary, games = evaluate_tree(tree, words, model_path, weights=weights)
        summary_rows.append(summary)
        hard_rows.extend(hardest_rows(length, games))

    summary_rows.sort(key=lambda row: int(row["length"]))
    hard_rows.sort(key=lambda row: (int(row["length"]), str(row.get("category", "")), str(row.get("word", ""))))
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "performance_by_length.csv", summary_rows)
    write_csv(
        output_dir / "best_first_letter_by_length.csv",
        [
            {
                "length": row["length"],
                "number_of_words": row["total_words"],
                "best_first_guess": row["best_first_guess"],
                "win_rate": row["win_rate"],
                "average_mistakes": row["average_wrong_guesses"],
                "training_strategy": row["training_strategy"],
            }
            for row in summary_rows
        ],
    )
    write_csv(
        output_dir / "tree_complexity.csv",
        [
            {
                "length": row["length"],
                "number_of_words": row["total_words"],
                "node_count": row["node_count"],
                "leaf_count": row["leaf_count"],
                "maximum_tree_depth": row["maximum_tree_depth"],
                "average_traversal_depth": row["average_traversal_depth"],
                "model_file_size": row["model_file_size"],
                "training_time": row["training_time"],
            }
            for row in summary_rows
        ],
    )
    write_csv(output_dir / "hardest_words_by_length.csv", hard_rows)

    figures = PROJECT_ROOT / "reports" / "figures"
    suffix = "" if args.weighting == "uniform" else f"_{args.weighting}"
    plot_metric(summary_rows, "win_rate", figures / f"win_rate_vs_word_length{suffix}.png", "Win rate")
    plot_metric(summary_rows, "average_wrong_guesses", figures / f"mistakes_vs_word_length{suffix}.png", "Average mistakes")
    plot_metric(summary_rows, "node_count", figures / f"tree_size_vs_word_length{suffix}.png", "Tree nodes")
    plot_metric(summary_rows, "total_words", figures / f"vocabulary_size_vs_word_length{suffix}.png", "Vocabulary size")
    plot_metric(summary_rows, "training_time", figures / f"training_time_vs_word_length{suffix}.png", "Training time (s)")

    if summary_rows:
        easiest = max(summary_rows, key=lambda row: (row["win_rate"], -row["average_wrong_guesses"]))
        hardest = min(summary_rows, key=lambda row: (row["win_rate"], -row["average_wrong_guesses"]))
        (output_dir / "strategy_difference_summary.json").write_text(
            json.dumps(
                {
                    "easiest_length": easiest,
                    "hardest_length": hardest,
                    "mean_win_rate": statistics.mean(float(row["win_rate"]) for row in summary_rows),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    logging.info("Wrote aggregate tree reports to %s", output_dir)


if __name__ == "__main__":
    main()
