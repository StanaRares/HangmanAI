from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.core.dictionary import PROJECT_ROOT  # noqa: E402
from app.tree.builder import HangmanTreeBuilder, TreeBuildConfig  # noqa: E402
from app.tree.evaluation import evaluate_tree  # noqa: E402
from app.tree.vocabulary import load_vocabulary_frame, weights_for_length, words_for_length  # noqa: E402
from train_tree import choose_profile, parse_lookahead  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all possible root letters for one word length.")
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--vocabulary", default=None)
    parser.add_argument("--weighting", choices=["uniform", "wordfreq"], default="uniform")
    parser.add_argument("--profile", choices=["auto", "fast", "balanced", "deep", "exact"], default="fast")
    parser.add_argument("--strategy", choices=["greedy", "optimized", "exact"], default="greedy")
    parser.add_argument("--lookahead", default=None)
    parser.add_argument("--node-budget", type=int, default=8000)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--pruning-min-gain", type=float, default=0.0)
    parser.add_argument("--max-words", type=int, default=None, help="Smoke-test limit. Omit for all words.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports" / "results" / "root_letter_analysis"))
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], path: Path, length: int) -> None:
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["win_rate"]),
            float(row["average_mistakes"]),
            float(row["average_total_guesses"]),
            int(row["node_count"]),
            str(row["root_letter"]),
        ),
    )
    plt.figure(figsize=(10, 5))
    plt.bar([row["root_letter"].upper() for row in ranked], [row["win_rate"] for row in ranked], color="#54d6a7")
    plt.xlabel("Forced root letter")
    plt.ylabel("Achievable subtree win rate")
    plt.title(f"Root Letter Analysis: {length}-Letter Words")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    frame = load_vocabulary_frame(args.vocabulary)
    words = words_for_length(frame, args.length)
    if args.max_words:
        words = words[: args.max_words]
    if not words:
        raise SystemExit(f"No words available for length {args.length}.")
    weights = weights_for_length(frame, args.length) if args.weighting == "wordfreq" else None
    profile = choose_profile(args.length, len(words), args.profile)
    lookahead = parse_lookahead(args.lookahead, args.length)
    strategy = "bounded-lookahead" if args.strategy == "optimized" else args.strategy
    rows = []
    for letter in "abcdefghijklmnopqrstuvwxyz":
        config = TreeBuildConfig.from_profile(
            length=args.length,
            profile=profile,  # type: ignore[arg-type]
            weighting=args.weighting,
            lookahead=lookahead,
            node_budget=args.node_budget,
        )
        config = TreeBuildConfig(**{**config.__dict__, "strategy": strategy})
        if args.max_depth is not None:
            config = TreeBuildConfig(**{**config.__dict__, "max_depth": args.max_depth})
        if args.pruning_min_gain:
            config = TreeBuildConfig(**{**config.__dict__, "pruning_min_gain": args.pruning_min_gain})
        logging.info("[%s letters] root %s", args.length, letter.upper())
        tree = HangmanTreeBuilder(words, weights, config).build(forced_root_letter=letter)
        summary, _ = evaluate_tree(tree, words, weights=weights)
        rows.append(
            {
                "length": args.length,
                "root_letter": letter,
                "words": len(words),
                "win_rate": summary["win_rate"],
                "weighted_win_rate": summary["weighted_win_rate"],
                "average_mistakes": summary["average_wrong_guesses"],
                "weighted_average_mistakes": summary["weighted_average_wrong_guesses"],
                "average_total_guesses": summary["average_total_guesses"],
                "node_count": tree.node_count,
                "maximum_depth": tree.max_depth,
                "training_strategy": tree.strategy,
                "weighting": args.weighting,
                "training_time": tree.metadata.get("training_seconds", 0.0),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["win_rate"]),
            float(row["average_mistakes"]),
            float(row["average_total_guesses"]),
            int(row["node_count"]),
            str(row["root_letter"]),
        )
    )
    output_dir = Path(args.output_dir)
    write_csv(output_dir / f"length_{args.length}_{args.weighting}.csv", rows)
    plot(rows, PROJECT_ROOT / "reports" / "figures" / f"root_letter_analysis_length_{args.length}_{args.weighting}.png", args.length)
    logging.info("Best root for length %s: %s", args.length, rows[0]["root_letter"].upper())


if __name__ == "__main__":
    main()
