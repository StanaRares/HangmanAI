from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.agents.frequency_agent import CandidateFrequencyAgent  # noqa: E402
from app.core.candidate_filter import CandidateIndex  # noqa: E402
from app.core.dictionary import PROJECT_ROOT, read_words  # noqa: E402
from app.core.environment import HangmanEnvironment  # noqa: E402
from app.ml.features import FeatureConfig, extract_feature_dict, feature_names  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decision Tree learning-curve experiment.")
    parser.add_argument(
        "--train-data",
        default=str(PROJECT_ROOT / "data" / "processed" / "training_states.parquet"),
    )
    parser.add_argument(
        "--validation-data",
        default=str(PROJECT_ROOT / "data" / "processed" / "validation_states.parquet"),
    )
    parser.add_argument(
        "--solver-dictionary",
        default=str(PROJECT_ROOT / "data" / "processed" / "vocabulary.txt"),
    )
    parser.add_argument(
        "--validation-words",
        default=str(PROJECT_ROOT / "data" / "processed" / "validation.txt"),
    )
    parser.add_argument("--game-counts", default="1000,5000,10000,25000,50000,100000")
    parser.add_argument("--feature-set", choices=["public_state", "candidate_stats", "full"], default="full")
    parser.add_argument("--max-validation-games", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports" / "results" / "learning_curve.csv"),
    )
    return parser.parse_args()


def read_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix == ".csv":
        return pd.read_csv(source)
    return pd.read_parquet(source)


def parse_counts(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def choose_letter(model, names: list[str], state, candidates: tuple[str, ...], index: CandidateIndex) -> str:
    feature_set = "full" if any(name.startswith("entropy_") for name in names) else "candidate_stats" if "candidate_count" in names else "public_state"
    feature_dict = extract_feature_dict(state, candidates, index, feature_set=feature_set)  # type: ignore[arg-type]
    vector = np.array([[feature_dict.get(name, 0.0) for name in names]])
    probabilities = model.predict_proba(vector)[0]
    rankings = sorted(
        (
            (str(letter), float(probability))
            for letter, probability in zip(model.classes_, probabilities, strict=False)
            if str(letter) not in state.guessed_letters
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if rankings and rankings[0][1] > 0:
        return rankings[0][0]
    return CandidateFrequencyAgent().guess(state, index).letter


def hangman_metrics(model, names: list[str], dictionary: list[str], words: list[str], max_games: int) -> dict[str, float]:
    selected_words = words[:max_games]
    if not selected_words:
        return {
            "hangman_win_rate": 0.0,
            "average_wrong_guesses": 0.0,
            "average_remaining_lives": 0.0,
            "average_turns": 0.0,
            "average_inference_time_ms": 0.0,
        }
    index = CandidateIndex(dictionary)
    wins = 0
    wrong: list[int] = []
    remaining: list[int] = []
    turns: list[int] = []
    timings: list[float] = []
    for word in selected_words:
        environment = HangmanEnvironment(dictionary or [word], max_lives=6)
        state = environment.reset(word)
        while not state.is_finished:
            candidates = index.candidates(state)
            started = time.perf_counter()
            letter = choose_letter(model, names, state, candidates, index)
            timings.append((time.perf_counter() - started) * 1000)
            state = environment.step(letter)
        wins += int(state.status == "won")
        wrong.append(len(state.incorrect_letters))
        remaining.append(state.remaining_lives)
        turns.append(state.turn)
    return {
        "hangman_win_rate": wins / len(selected_words),
        "average_wrong_guesses": float(np.mean(wrong)),
        "average_remaining_lives": float(np.mean(remaining)),
        "average_turns": float(np.mean(turns)),
        "average_inference_time_ms": float(np.mean(timings)) if timings else 0.0,
    }


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    train = read_frame(args.train_data)
    validation = read_frame(args.validation_data)
    dictionary = read_words(args.solver_dictionary)
    validation_words = read_words(args.validation_words)
    names = feature_names(FeatureConfig(), args.feature_set)

    if "game" not in train.columns:
        raise SystemExit("Training data needs a game column for learning curves.")

    rows = []
    for game_count in parse_counts(args.game_counts):
        subset = train[train["game"] < game_count]
        if subset.empty:
            logging.warning("Skipping %s games because no rows are available.", game_count)
            continue
        model = DecisionTreeClassifier(
            criterion="gini",
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=args.random_state,
        )
        started = time.perf_counter()
        model.fit(subset[names].to_numpy(), subset["label"].to_numpy())
        training_seconds = time.perf_counter() - started
        action_accuracy = float(
            accuracy_score(validation["label"].to_numpy(), model.predict(validation[names].to_numpy()))
        )
        live = hangman_metrics(
            model,
            names,
            dictionary,
            validation_words,
            args.max_validation_games,
        )
        row = {
            "model": "DecisionTreeClassifier",
            "feature_set": args.feature_set,
            "training_games": game_count,
            "training_rows": len(subset),
            "action_prediction_accuracy": round(action_accuracy, 6),
            "hangman_win_rate": round(live["hangman_win_rate"], 6),
            "average_wrong_guesses": round(live["average_wrong_guesses"], 6),
            "average_remaining_lives": round(live["average_remaining_lives"], 6),
            "average_turns": round(live["average_turns"], 6),
            "average_inference_time_ms": round(live["average_inference_time_ms"], 6),
            "training_seconds": round(training_seconds, 6),
            "tree_depth": model.get_depth(),
            "node_count": model.tree_.node_count,
        }
        rows.append(row)
        logging.info("Learning-curve point: %s", row)

    if rows:
        write_rows(args.output, rows)
        logging.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
