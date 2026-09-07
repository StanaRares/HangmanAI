from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
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
    parser = argparse.ArgumentParser(description="Train and tune a DecisionTree Hangman agent.")
    parser.add_argument(
        "--train-data",
        default=str(PROJECT_ROOT / "data" / "processed" / "training_states.parquet"),
    )
    parser.add_argument("--validation-data", default=None)
    parser.add_argument(
        "--dictionary",
        default=str(PROJECT_ROOT / "data" / "processed" / "train.txt"),
        help="Solver dictionary for live Hangman validation.",
    )
    parser.add_argument(
        "--validation-words",
        default=str(PROJECT_ROOT / "data" / "processed" / "validation.txt"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "models" / "decision_tree.joblib"),
    )
    parser.add_argument(
        "--results",
        default=str(PROJECT_ROOT / "reports" / "results" / "decision_tree_tuning.csv"),
    )
    parser.add_argument("--max-validation-games", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def read_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix == ".csv":
        return pd.read_csv(source)
    return pd.read_parquet(source)


def fit_model(
    train: pd.DataFrame,
    features: list[str],
    criterion: str,
    max_depth: int | None,
    min_samples_leaf: int,
    random_state: int,
) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=2,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(train[features].to_numpy(), train["label"].to_numpy())
    return model


def action_accuracy(model: DecisionTreeClassifier, validation: pd.DataFrame, features: list[str]) -> float:
    if validation.empty:
        return 0.0
    predictions = model.predict(validation[features].to_numpy())
    return float(accuracy_score(validation["label"].to_numpy(), predictions))


def choose_letter(
    model: DecisionTreeClassifier,
    feature_names_for_model: list[str],
    state,
    candidates: tuple[str, ...],
    index: CandidateIndex,
) -> str:
    feature_dict = extract_feature_dict(state, candidates, index)
    vector = np.array([[feature_dict.get(name, 0.0) for name in feature_names_for_model]])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        ranking = sorted(
            (
                (str(letter), float(probability))
                for letter, probability in zip(model.classes_, probabilities, strict=False)
                if str(letter) not in state.guessed_letters
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if ranking and ranking[0][1] > 0:
            return ranking[0][0]
    prediction = str(model.predict(vector)[0])
    if prediction not in state.guessed_letters:
        return prediction
    return CandidateFrequencyAgent().guess(state, index).letter


def hangman_validation_metrics(
    model: DecisionTreeClassifier,
    feature_names_for_model: list[str],
    dictionary: list[str],
    validation_words: list[str],
    max_games: int,
) -> dict[str, float]:
    if not validation_words:
        return {"win_rate": 0.0, "avg_wrong_guesses": 0.0}
    index = CandidateIndex(dictionary)
    selected_words = validation_words[:max_games]
    wins = 0
    wrong_guesses = []
    for word in selected_words:
        environment = HangmanEnvironment(dictionary or [word], max_lives=6)
        state = environment.reset(word)
        while not state.is_finished:
            candidates = index.candidates(state)
            letter = choose_letter(model, feature_names_for_model, state, candidates, index)
            state = environment.step(letter)
        wins += int(state.status == "won")
        wrong_guesses.append(len(state.incorrect_letters))
    return {
        "win_rate": wins / len(selected_words),
        "avg_wrong_guesses": float(np.mean(wrong_guesses)) if wrong_guesses else 0.0,
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
    validation = read_frame(args.validation_data) if args.validation_data else pd.DataFrame()
    features = feature_names(FeatureConfig())

    missing = [name for name in features + ["label"] if name not in train.columns]
    if missing:
        raise SystemExit(f"Training data is missing columns: {missing[:10]}")

    dictionary = read_words(args.dictionary)
    validation_words = read_words(args.validation_words) if Path(args.validation_words).exists() else []
    grid = [
        ("gini", 6, 1),
        ("gini", 10, 1),
        ("gini", 14, 2),
        ("entropy", 6, 1),
        ("entropy", 10, 2),
        ("entropy", 14, 5),
        ("log_loss", 10, 2),
        ("gini", None, 5),
    ]

    rows: list[dict[str, Any]] = []
    best: tuple[float, float, DecisionTreeClassifier, dict[str, Any]] | None = None
    for criterion, max_depth, min_samples_leaf in grid:
        model = fit_model(
            train,
            features,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=args.random_state,
        )
        accuracy = action_accuracy(model, validation, features) if not validation.empty else 0.0
        live_metrics = hangman_validation_metrics(
            model,
            features,
            dictionary,
            validation_words,
            max_games=args.max_validation_games,
        )
        row = {
            "model": "DecisionTreeClassifier",
            "criterion": criterion,
            "max_depth": "None" if max_depth is None else max_depth,
            "min_samples_leaf": min_samples_leaf,
            "validation_action_accuracy": round(accuracy, 6),
            "hangman_validation_win_rate": round(live_metrics["win_rate"], 6),
            "average_wrong_guesses": round(live_metrics["avg_wrong_guesses"], 6),
            "tree_depth": model.get_depth(),
            "node_count": model.tree_.node_count,
        }
        rows.append(row)
        score = (live_metrics["win_rate"], -live_metrics["avg_wrong_guesses"])
        if best is None or score > (best[0], best[1]):
            best = (score[0], score[1], model, row)
        logging.info("Tuned %s", row)

    if best is None:
        raise SystemExit("No model was trained.")

    write_rows(args.results, rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best[2],
            "feature_names": features,
            "feature_config": {"max_word_length": FeatureConfig().max_word_length},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_rows": len(train),
            "best_validation": best[3],
        },
        output,
    )
    logging.info("Saved best model to %s with metrics %s", output, best[3])


if __name__ == "__main__":
    main()
