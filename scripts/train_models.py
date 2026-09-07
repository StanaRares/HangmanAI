from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.agents.frequency_agent import CandidateFrequencyAgent  # noqa: E402
from app.core.candidate_filter import CandidateIndex  # noqa: E402
from app.core.dictionary import PROJECT_ROOT, read_words  # noqa: E402
from app.core.environment import HangmanEnvironment  # noqa: E402
from app.ml.features import FeatureConfig, FeatureSet, extract_feature_dict, feature_names  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

MODEL_FAMILY_ALIASES = {
    "decision_tree": "decision_tree",
    "random_forest": "random_forest",
    "hist_gradient_boosting": "boosting",
    "boosting": "boosting",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ablation and tree-family Hangman next-letter models."
    )
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
        help="Candidate dictionary used for live validation. Should include validation/test words for generalization.",
    )
    parser.add_argument(
        "--validation-words",
        default=str(PROJECT_ROOT / "data" / "processed" / "validation.txt"),
    )
    parser.add_argument(
        "--test-words-for-leakage-check",
        default=str(PROJECT_ROOT / "data" / "processed" / "test.txt"),
    )
    parser.add_argument(
        "--feature-sets",
        default="public_state,candidate_stats,full",
        help="Comma-separated feature sets: public_state,candidate_stats,full.",
    )
    parser.add_argument(
        "--model-families",
        default="decision_tree,random_forest,hist_gradient_boosting",
        help="Comma-separated model families.",
    )
    parser.add_argument("--max-validation-games", type=int, default=200)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--random-forest-estimators", type=int, default=80)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "models"))
    parser.add_argument(
        "--results",
        default=str(PROJECT_ROOT / "reports" / "results" / "model_experiments.csv"),
    )
    parser.add_argument(
        "--importance-output",
        default=str(PROJECT_ROOT / "reports" / "figures" / "model_feature_importances.csv"),
    )
    return parser.parse_args()


def read_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix == ".csv":
        return pd.read_csv(source)
    return pd.read_parquet(source)


def parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def make_model(family: str, args: argparse.Namespace):
    canonical = MODEL_FAMILY_ALIASES.get(family)
    if canonical == "decision_tree":
        return DecisionTreeClassifier(
            criterion="gini",
            max_depth=14,
            min_samples_leaf=2,
            min_samples_split=2,
            class_weight="balanced",
            random_state=args.random_state,
        )
    if canonical == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.random_forest_estimators,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=args.random_state,
        )
    if canonical == "boosting":
        return HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=args.random_state,
        )
    raise ValueError(f"Unsupported model family: {family}")


def feature_set_from_names(names: list[str]) -> str:
    if any(name.startswith("entropy_") for name in names):
        return "full"
    if "candidate_count" in names:
        return "candidate_stats"
    return "public_state"


def choose_letter(model, names: list[str], state, candidates: tuple[str, ...], index: CandidateIndex) -> str:
    feature_dict = extract_feature_dict(
        state,
        candidates,
        index,
        feature_set=feature_set_from_names(names),  # type: ignore[arg-type]
    )
    vector = np.array([[feature_dict.get(name, 0.0) for name in names]])
    if hasattr(model, "predict_proba"):
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
    prediction = str(model.predict(vector)[0])
    if prediction not in state.guessed_letters:
        return prediction
    return CandidateFrequencyAgent().guess(state, index).letter


def hangman_metrics(
    model,
    names: list[str],
    dictionary: list[str],
    validation_words: list[str],
    max_games: int,
    max_lives: int = 6,
) -> dict[str, float]:
    selected_words = validation_words[:max_games]
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
    remaining_lives: list[int] = []
    turns: list[int] = []
    timings: list[float] = []
    for word in selected_words:
        environment = HangmanEnvironment(dictionary or [word], max_lives=max_lives)
        state = environment.reset(word)
        while not state.is_finished:
            candidates = index.candidates(state)
            started = time.perf_counter()
            letter = choose_letter(model, names, state, candidates, index)
            timings.append((time.perf_counter() - started) * 1000)
            state = environment.step(letter)
        wins += int(state.status == "won")
        wrong.append(len(state.incorrect_letters))
        remaining_lives.append(state.remaining_lives)
        turns.append(state.turn)
    return {
        "hangman_win_rate": wins / len(selected_words),
        "average_wrong_guesses": float(np.mean(wrong)),
        "average_remaining_lives": float(np.mean(remaining_lives)),
        "average_turns": float(np.mean(turns)),
        "average_inference_time_ms": float(np.mean(timings)) if timings else 0.0,
    }


def leakage_audit(train: pd.DataFrame, test_words: list[str]) -> dict[str, Any]:
    train_words = set(train["hidden_word"].dropna().astype(str)) if "hidden_word" in train.columns else set()
    overlap = sorted(train_words.intersection(test_words))
    return {
        "training_hidden_word_count": len(train_words),
        "test_word_count": len(test_words),
        "overlap_count": len(overlap),
        "overlap_examples": overlap[:25],
        "has_hidden_word_provenance": "hidden_word" in train.columns,
    }


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_stats(model) -> dict[str, Any]:
    return {
        "tree_depth": model.get_depth() if hasattr(model, "get_depth") else "",
        "node_count": model.tree_.node_count if hasattr(model, "tree_") else "",
        "estimators": len(model.estimators_) if hasattr(model, "estimators_") else "",
    }


def importance_rows(model, names: list[str], model_id: str, feature_set: str) -> list[dict[str, Any]]:
    if not hasattr(model, "feature_importances_"):
        return []
    importances = getattr(model, "feature_importances_")
    rows = [
        {
            "model_id": model_id,
            "feature_set": feature_set,
            "feature": name,
            "importance": float(importance),
        }
        for name, importance in zip(names, importances, strict=False)
        if importance > 0
    ]
    return sorted(rows, key=lambda row: row["importance"], reverse=True)


def save_model(path: Path, model, names: list[str], feature_set: str, family: str, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": names,
            "feature_config": {"max_word_length": FeatureConfig().max_word_length},
            "feature_set": feature_set,
            "model_family": family,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_rows": row["training_rows"],
            "validation": row,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    train = read_frame(args.train_data)
    validation = read_frame(args.validation_data)
    if args.max_train_rows:
        train = train.head(args.max_train_rows)

    dictionary = read_words(args.solver_dictionary)
    validation_words = read_words(args.validation_words) if Path(args.validation_words).exists() else []
    test_words = (
        read_words(args.test_words_for_leakage_check)
        if Path(args.test_words_for_leakage_check).exists()
        else []
    )
    audit = leakage_audit(train, test_words)
    if audit["overlap_count"]:
        raise SystemExit(
            "Training data contains hidden test words; refusing to train. "
            f"Examples: {audit['overlap_examples']}"
        )

    output_dir = Path(args.output_dir)
    feature_sets = parse_csv_arg(args.feature_sets)
    families = parse_csv_arg(args.model_families)
    result_rows: list[dict[str, Any]] = []
    importances: list[dict[str, Any]] = []
    best_by_family: dict[str, tuple[float, float, Path]] = {}

    for feature_set in feature_sets:
        names = feature_names(FeatureConfig(), feature_set)  # type: ignore[arg-type]
        missing = [name for name in names + ["label"] if name not in train.columns]
        if missing:
            raise SystemExit(f"Dataset missing columns for {feature_set}: {missing[:10]}")
        x_train = train[names].to_numpy()
        y_train = train["label"].to_numpy()
        x_val = validation[names].to_numpy() if not validation.empty else np.empty((0, len(names)))
        y_val = validation["label"].to_numpy() if not validation.empty else np.array([])

        for family in families:
            canonical = MODEL_FAMILY_ALIASES.get(family, family)
            model_id = f"{canonical}_{feature_set}"
            logging.info("Training %s on %s rows with %s features", model_id, len(train), len(names))
            started = time.perf_counter()
            model = make_model(family, args)
            model.fit(x_train, y_train)
            training_seconds = time.perf_counter() - started
            accuracy = (
                float(accuracy_score(y_val, model.predict(x_val))) if len(y_val) else 0.0
            )
            live = hangman_metrics(
                model,
                names,
                dictionary,
                validation_words,
                args.max_validation_games,
            )
            row = {
                "model_id": model_id,
                "model_family": canonical,
                "feature_set": feature_set,
                "training_rows": len(train),
                "feature_count": len(names),
                "validation_action_accuracy": round(accuracy, 6),
                "hangman_win_rate": round(live["hangman_win_rate"], 6),
                "average_wrong_guesses": round(live["average_wrong_guesses"], 6),
                "average_remaining_lives": round(live["average_remaining_lives"], 6),
                "average_turns": round(live["average_turns"], 6),
                "average_inference_time_ms": round(live["average_inference_time_ms"], 6),
                "training_seconds": round(training_seconds, 6),
                "leakage_overlap_count": audit["overlap_count"],
                **model_stats(model),
            }
            model_path = output_dir / f"{model_id}.joblib"
            save_model(model_path, model, names, feature_set, canonical, row)
            result_rows.append(row)
            importances.extend(importance_rows(model, names, model_id, feature_set)[:50])
            logging.info("Finished %s: %s", model_id, row)

            score = (live["hangman_win_rate"], -live["average_wrong_guesses"])
            if canonical not in best_by_family or score > (
                best_by_family[canonical][0],
                best_by_family[canonical][1],
            ):
                best_by_family[canonical] = (score[0], score[1], model_path)

    write_rows(args.results, result_rows)
    write_rows(args.importance_output, importances)
    for family, (_, _, model_path) in best_by_family.items():
        alias = output_dir / f"{family}.joblib"
        shutil.copyfile(model_path, alias)
        logging.info("Copied best %s model to %s", family, alias)


if __name__ == "__main__":
    main()
