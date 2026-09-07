from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.core.dictionary import PROJECT_ROOT  # noqa: E402
from app.tree.vocabulary import load_vocabulary_frame  # noqa: E402
from train_tree import train_tree  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Hangman decision trees for every available word length.")
    parser.add_argument("--vocabulary", default=None)
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--min-length", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-lives", type=int, default=6)
    parser.add_argument("--weighting", choices=["uniform", "wordfreq"], default="uniform")
    parser.add_argument("--profile", choices=["auto", "fast", "balanced", "deep", "exact"], default="auto")
    parser.add_argument("--strategy", choices=["auto", "greedy", "optimized", "exact"], default="auto")
    parser.add_argument("--lookahead", default=None)
    parser.add_argument("--node-budget", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--pruning-min-gain", type=float, default=0.0)
    parser.add_argument("--max-words", type=int, default=None, help="Smoke-test limit per length. Omit for all words.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reports" / "results" / "train_all_trees.csv"))
    return parser.parse_args()


def available_lengths(args: argparse.Namespace) -> list[int]:
    frame = load_vocabulary_frame(args.vocabulary)
    lengths = sorted(int(length) for length in frame["length"].unique())
    if args.min_length is not None:
        lengths = [length for length in lengths if length >= args.min_length]
    if args.max_length is not None:
        lengths = [length for length in lengths if length <= args.max_length]
    return lengths


def args_for_length(args: argparse.Namespace, length: int) -> argparse.Namespace:
    return argparse.Namespace(
        length=length,
        vocabulary=args.vocabulary,
        models_dir=args.models_dir,
        max_lives=args.max_lives,
        weighting=args.weighting,
        profile=args.profile,
        strategy=args.strategy,
        lookahead=args.lookahead,
        node_budget=args.node_budget,
        max_depth=args.max_depth,
        pruning_min_gain=args.pruning_min_gain,
        max_words=args.max_words,
        no_evaluate=args.no_evaluate,
    )


def is_complete(args: argparse.Namespace, length: int) -> bool:
    return (Path(args.models_dir) / f"length_{length}" / args.weighting / "training_statistics.json").exists()


def run_length(length_args: argparse.Namespace) -> dict[str, Any]:
    return train_tree(length_args)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_checkpoint(path: str | Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows)
    checkpoint = Path(path).with_suffix(".json")
    checkpoint.write_text(
        json.dumps({"completed_lengths": [row["length"] for row in rows]}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    lengths = available_lengths(args)
    jobs = []
    for length in lengths:
        if args.resume and is_complete(args, length):
            logging.info("[%s letters] Skipping completed tree.", length)
            continue
        jobs.append(args_for_length(args, length))

    results = []
    if args.workers <= 1:
        for job in jobs:
            results.append(run_length(job))
            if len(results) % max(args.checkpoint_every, 1) == 0:
                write_checkpoint(args.output, results)
    else:
        # Word lengths are independent. Keep workers conservative to avoid loading
        # many large vocabulary buckets at once.
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_length, job): job.length for job in jobs}
            for future in as_completed(futures):
                length = futures[future]
                try:
                    results.append(future.result())
                    logging.info("[%s letters] completed.", length)
                    if len(results) % max(args.checkpoint_every, 1) == 0:
                        write_checkpoint(args.output, results)
                except Exception:
                    logging.exception("[%s letters] failed.", length)

    write_checkpoint(args.output, results)
    logging.info("Wrote training summary to %s", args.output)


if __name__ == "__main__":
    main()
