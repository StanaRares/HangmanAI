from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.dictionary import PROJECT_ROOT  # noqa: E402
from app.tree.builder import HangmanTreeBuilder, TreeBuildConfig  # noqa: E402
from app.tree.evaluation import evaluate_tree  # noqa: E402
from app.tree.serialization import save_tree  # noqa: E402
from app.tree.vocabulary import load_vocabulary_frame, weights_for_length, words_for_length  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one word-length-specific Hangman decision tree.")
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--vocabulary", default=None)
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--max-lives", type=int, default=6)
    parser.add_argument("--weighting", choices=["uniform", "wordfreq"], default="uniform")
    parser.add_argument("--profile", choices=["auto", "fast", "balanced", "deep", "exact"], default="auto")
    parser.add_argument("--strategy", choices=["auto", "greedy", "optimized", "exact"], default="auto")
    parser.add_argument("--lookahead", default=None, help="Integer depth or 'full'.")
    parser.add_argument("--node-budget", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--pruning-min-gain", type=float, default=0.0)
    parser.add_argument("--max-words", type=int, default=None, help="Smoke-test limit. Omit for all words.")
    parser.add_argument("--no-evaluate", action="store_true")
    return parser.parse_args()


def parse_lookahead(value: str | None, length: int) -> int | None:
    if value is None:
        return None
    if value == "full":
        return 26
    return int(value)


def choose_profile(length: int, word_count: int, requested: str) -> str:
    if requested != "auto":
        return requested
    if length <= 3 and word_count <= 500:
        return "exact"
    if length <= 4 and word_count <= 2000:
        return "deep"
    if word_count <= 8000:
        return "balanced"
    return "fast"


def config_from_args(args: argparse.Namespace, word_count: int) -> TreeBuildConfig:
    profile = choose_profile(args.length, word_count, args.profile)
    lookahead = parse_lookahead(args.lookahead, args.length)
    config = TreeBuildConfig.from_profile(
        length=args.length,
        profile=profile,  # type: ignore[arg-type]
        max_lives=args.max_lives,
        weighting=args.weighting,
        lookahead=lookahead,
        node_budget=args.node_budget,
    )
    if args.strategy != "auto":
        strategy = "bounded-lookahead" if args.strategy == "optimized" else args.strategy
        config = TreeBuildConfig(
            **{
                **config.__dict__,
                "strategy": strategy,
            }
        )
    if args.max_depth is not None:
        config = TreeBuildConfig(**{**config.__dict__, "max_depth": args.max_depth})
    if args.pruning_min_gain:
        config = TreeBuildConfig(**{**config.__dict__, "pruning_min_gain": args.pruning_min_gain})
    return config


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def archive_extensions(model_dir: Path) -> None:
    extension_dir = model_dir / "extensions"
    if not extension_dir.exists():
        return
    if not any(extension_dir.iterdir()):
        extension_dir.rmdir()
        return
    archive = model_dir / f"extensions_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.move(str(extension_dir), str(archive))
    logging.info("Archived stale extension files to %s", archive)


def train_tree(args: argparse.Namespace) -> dict[str, Any]:
    frame = load_vocabulary_frame(args.vocabulary)
    words = words_for_length(frame, args.length)
    if args.max_words:
        words = words[: args.max_words]
    if not words:
        raise SystemExit(f"No words available for length {args.length}.")
    weights = weights_for_length(frame, args.length) if args.weighting == "wordfreq" else None
    config = config_from_args(args, len(words))
    logging.info(
        "[%s letters] %s words. Building %s/%s tree...",
        args.length,
        len(words),
        config.profile,
        config.strategy,
    )
    builder = HangmanTreeBuilder(words, weights, config)
    tree = builder.build()
    model_dir = Path(args.models_dir) / f"length_{args.length}" / args.weighting
    archive_extensions(model_dir)
    model_path = model_dir / "tree.json.gz"
    save_tree(tree, model_path)
    vocabulary_stats = {
        "length": args.length,
        "words": len(words),
        "weighting": args.weighting,
        "source_vocabulary": args.vocabulary or "default processed wordfreq vocabulary",
        "max_words_smoke_limit": args.max_words,
    }
    write_json(model_dir / "metadata.json", tree.metadata)
    write_json(model_dir / "vocabulary_stats.json", vocabulary_stats)
    result: dict[str, Any] = {
        "length": args.length,
        "words": len(words),
        "model_path": str(model_path),
        "best_first_guess": tree.best_first_guess,
        "node_count": tree.node_count,
        "leaf_count": tree.leaf_count,
        "max_depth": tree.max_depth,
        "strategy": tree.strategy,
        "weighting": args.weighting,
        "training_time": tree.metadata.get("training_seconds", 0.0),
    }
    if not args.no_evaluate:
        summary, games = evaluate_tree(tree, words, model_path, weights=weights)
        reports_dir = PROJECT_ROOT / "reports" / "results" / "trees"
        reports_dir.mkdir(parents=True, exist_ok=True)
        import csv

        game_path = reports_dir / f"length_{args.length}_{args.weighting}_games.csv"
        with game_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(games[0]))
            writer.writeheader()
            writer.writerows(games)
        write_json(model_dir / "evaluation_results.json", summary)
        result.update(summary)
    write_json(model_dir / "training_statistics.json", result)
    logging.info(
        "[%s letters] Done. first=%s win_rate=%s nodes=%s depth=%s",
        args.length,
        result.get("best_first_guess"),
        round(float(result.get("win_rate", 0.0)), 4),
        result["node_count"],
        result["max_depth"],
    )
    return result


def main() -> None:
    args = parse_args()
    train_tree(args)


if __name__ == "__main__":
    main()
