from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.agents import (  # noqa: E402
    BoostingAgent,
    CandidateFrequencyAgent,
    DecisionTreeAgent,
    EntropyAgent,
    GlobalFrequencyAgent,
    RandomAgent,
    RandomForestAgent,
)
from app.core.candidate_filter import CandidateIndex  # noqa: E402
from app.core.dictionary import PROJECT_ROOT, read_words  # noqa: E402
from app.core.environment import HangmanEnvironment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark HangmanAI agents on identical words.")
    parser.add_argument(
        "--evaluation-mode",
        choices=["known_word", "model_generalization", "oov", "custom"],
        default="model_generalization",
        help=(
            "known_word: hidden words are in the solver dictionary. "
            "model_generalization: model training words are disjoint from hidden test words, "
            "but solver dictionary contains the test words. "
            "oov: hidden words are absent from the solver dictionary. "
            "custom: use --dictionary directly."
        ),
    )
    parser.add_argument("--dictionary", default=None, help="Legacy/custom solver dictionary path.")
    parser.add_argument("--solver-dictionary", default=None, help="Override the mode-derived solver dictionary.")
    parser.add_argument("--train-words", default=str(PROJECT_ROOT / "data" / "processed" / "train.txt"))
    parser.add_argument("--validation-words", default=str(PROJECT_ROOT / "data" / "processed" / "validation.txt"))
    parser.add_argument("--test-words", default=str(PROJECT_ROOT / "data" / "processed" / "test.txt"))
    parser.add_argument("--all-words", default=str(PROJECT_ROOT / "data" / "processed" / "vocabulary.txt"))
    parser.add_argument("--games", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-lives", type=int, default=6)
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--decision-tree-model", default=None)
    parser.add_argument("--random-forest-model", default=None)
    parser.add_argument("--boosting-model", default=None)
    parser.add_argument(
        "--agents",
        default="random,global_frequency,candidate_frequency,entropy,risk_adjusted_entropy,decision_tree,random_forest,boosting",
        help="Comma-separated agent names.",
    )
    parser.add_argument("--include-untrained-ml", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def union_words(*word_lists: list[str]) -> list[str]:
    return sorted(set().union(*word_lists))


def read_if_exists(path: str | Path) -> list[str]:
    source = Path(path)
    return read_words(source) if source.exists() else []


def mode_description(mode: str) -> str:
    return {
        "known_word": "Hidden test words exist in the solver candidate dictionary.",
        "model_generalization": (
            "Decision-tree training words/states exclude test words, while the solver "
            "candidate dictionary includes those test words."
        ),
        "oov": "Hidden test words are absent from the solver candidate dictionary.",
        "custom": "User-provided benchmark dictionary/test-word setup.",
    }[mode]


def resolve_benchmark_words(args: argparse.Namespace) -> tuple[list[str], list[str], dict[str, Any]]:
    train_words = read_if_exists(args.train_words)
    validation_words = read_if_exists(args.validation_words)
    test_words = read_if_exists(args.test_words)
    all_words = read_if_exists(args.all_words) or union_words(train_words, validation_words, test_words)
    if not test_words:
        raise SystemExit("No test words are available.")

    if args.solver_dictionary:
        solver_words = read_words(args.solver_dictionary)
    elif args.evaluation_mode in {"known_word", "model_generalization"}:
        solver_words = all_words
    elif args.evaluation_mode == "oov":
        solver_words = train_words
    elif args.evaluation_mode == "custom":
        if not args.dictionary:
            raise SystemExit("custom mode requires --dictionary.")
        solver_words = read_words(args.dictionary)
    else:
        raise SystemExit(f"Unknown evaluation mode: {args.evaluation_mode}")

    solver_set = set(solver_words)
    train_set = set(train_words)
    test_set = set(test_words)
    hidden_in_solver = sorted(test_set.intersection(solver_set))
    hidden_missing_from_solver = sorted(test_set.difference(solver_set))
    train_test_overlap = sorted(train_set.intersection(test_set))

    if args.evaluation_mode in {"known_word", "model_generalization"} and hidden_missing_from_solver:
        raise SystemExit(
            f"{args.evaluation_mode} requires test words in the solver dictionary. "
            f"Missing examples: {hidden_missing_from_solver[:10]}"
        )
    if args.evaluation_mode == "oov" and hidden_in_solver:
        raise SystemExit(
            "oov mode requires test words to be absent from the solver dictionary. "
            f"Overlapping examples: {hidden_in_solver[:10]}"
        )
    if args.evaluation_mode == "model_generalization" and train_test_overlap:
        raise SystemExit(
            "model_generalization requires train/test word splits to be disjoint. "
            f"Overlapping examples: {train_test_overlap[:10]}"
        )

    rng = random.Random(args.seed)
    selected_words = list(test_words)
    rng.shuffle(selected_words)
    selected_words = selected_words[: min(args.games, len(selected_words))]
    metadata = {
        "evaluation_mode": args.evaluation_mode,
        "mode_description": mode_description(args.evaluation_mode),
        "solver_dictionary_size": len(solver_words),
        "train_word_count": len(train_words),
        "validation_word_count": len(validation_words),
        "test_word_count": len(test_words),
        "evaluated_games": len(selected_words),
        "test_words_in_solver_dictionary": len(hidden_in_solver),
        "test_words_absent_from_solver_dictionary": len(hidden_missing_from_solver),
        "train_test_overlap_count": len(train_test_overlap),
        "seed": args.seed,
    }
    return solver_words, selected_words, metadata


def model_paths(args: argparse.Namespace) -> dict[str, Path]:
    models_dir = Path(args.models_dir)
    return {
        "decision_tree": Path(args.decision_tree_model)
        if args.decision_tree_model
        else models_dir / "decision_tree.joblib",
        "random_forest": Path(args.random_forest_model)
        if args.random_forest_model
        else models_dir / "random_forest.joblib",
        "boosting": Path(args.boosting_model)
        if args.boosting_model
        else models_dir / "boosting.joblib",
    }


def build_agents(args: argparse.Namespace, dictionary: list[str]) -> list[Any]:
    requested = set(parse_csv_arg(args.agents))
    paths = model_paths(args)
    agent_map = {
        "random": RandomAgent(seed=args.seed),
        "global_frequency": GlobalFrequencyAgent(dictionary),
        "candidate_frequency": CandidateFrequencyAgent(),
        "entropy": EntropyAgent(scoring="entropy_only", name="entropy"),
        "risk_adjusted_entropy": EntropyAgent(scoring="risk_adjusted_entropy"),
        "decision_tree": DecisionTreeAgent(paths["decision_tree"]),
        "random_forest": RandomForestAgent(paths["random_forest"]),
        "boosting": BoostingAgent(paths["boosting"]),
    }
    output = []
    for name, agent in agent_map.items():
        if name not in requested:
            continue
        if name in {"random_forest", "boosting"} and not paths[name].exists() and not args.include_untrained_ml:
            logging.info("Skipping %s because %s does not exist.", name, paths[name])
            continue
        output.append(agent)
    return output


def wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = wins / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * ((p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def word_characteristics(word: str, global_counts: Counter[str], dictionary_size: int) -> dict[str, Any]:
    repeated_letter_count = sum(count - 1 for count in Counter(word).values() if count > 1)
    rare_letters = {
        letter
        for letter in set(word)
        if dictionary_size and global_counts[letter] / dictionary_size <= 0.08
    }
    return {
        "has_repeated_letters": repeated_letter_count > 0,
        "repeated_letter_count": repeated_letter_count,
        "rare_letter_count": len(rare_letters),
        "rare_letters": "".join(sorted(rare_letters)),
    }


def candidate_bucket(count: int) -> str:
    if count <= 10:
        return "0000-0010"
    if count <= 100:
        return "0011-0100"
    if count <= 1000:
        return "0101-1000"
    if count <= 10000:
        return "1001-10000"
    return "10000+"


def play_game(
    agent,
    index: CandidateIndex,
    dictionary: list[str],
    word: str,
    game_index: int,
    max_lives: int,
    evaluation_mode: str,
    global_counts: Counter[str],
) -> dict[str, Any]:
    environment = HangmanEnvironment(dictionary or [word], max_lives=max_lives)
    state = environment.reset(word)
    decision_times: list[float] = []
    candidate_counts: list[int] = [len(index.candidates(state))]
    guesses: list[str] = []

    while not state.is_finished:
        started = time.perf_counter()
        decision = agent.guess(state, index)
        decision_times.append((time.perf_counter() - started) * 1000)
        guesses.append(decision.letter)
        state = environment.step(decision.letter)
        if state.status == "won":
            candidate_counts.append(1)
        elif state.status == "lost":
            candidate_counts.append(0)
        else:
            candidate_counts.append(len(index.candidates(state)))

    reductions = [
        max(candidate_counts[position] - candidate_counts[position + 1], 0)
        for position in range(len(candidate_counts) - 1)
    ]
    initial_candidates = candidate_counts[0] if candidate_counts else 0
    characteristics = word_characteristics(word, global_counts, len(dictionary))
    return {
        "evaluation_mode": evaluation_mode,
        "agent": agent.name,
        "game_index": game_index,
        "word": word,
        "word_length": len(word),
        "status": state.status,
        "won": state.status == "won",
        "guesses": len(state.guessed_letters),
        "guess_sequence": " ".join(guesses),
        "incorrect_guesses": len(state.incorrect_letters),
        "remaining_lives": state.remaining_lives,
        "turns": state.turn,
        "initial_candidate_count": initial_candidates,
        "candidate_size_bucket": candidate_bucket(initial_candidates),
        "candidate_counts_by_turn": json.dumps(candidate_counts),
        "avg_time_per_guess_ms": statistics.mean(decision_times) if decision_times else 0.0,
        "avg_candidate_reduction": statistics.mean(reductions) if reductions else 0.0,
        "initial_word_frequency": "",
        **characteristics,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_agent[row["agent"]].append(row)

    summaries = []
    for agent, agent_rows in by_agent.items():
        wins = sum(1 for row in agent_rows if row["won"])
        total = len(agent_rows)
        low, high = wilson_interval(wins, total)
        winning_rows = [row for row in agent_rows if row["won"]]
        summaries.append(
            {
                "evaluation_mode": agent_rows[0]["evaluation_mode"],
                "agent": agent,
                "games": total,
                "wins": wins,
                "losses": total - wins,
                "win_rate": round(wins / total if total else 0.0, 6),
                "loss_rate": round((total - wins) / total if total else 0.0, 6),
                "win_rate_ci_low": round(low, 6),
                "win_rate_ci_high": round(high, 6),
                "average_guesses": round(statistics.mean(row["guesses"] for row in agent_rows), 6),
                "average_incorrect_guesses": round(
                    statistics.mean(row["incorrect_guesses"] for row in agent_rows),
                    6,
                ),
                "average_remaining_lives": round(
                    statistics.mean(row["remaining_lives"] for row in agent_rows),
                    6,
                ),
                "average_remaining_lives_when_winning": round(
                    statistics.mean(row["remaining_lives"] for row in winning_rows) if winning_rows else 0.0,
                    6,
                ),
                "average_turns": round(statistics.mean(row["turns"] for row in agent_rows), 6),
                "average_decision_time_ms": round(
                    statistics.mean(row["avg_time_per_guess_ms"] for row in agent_rows),
                    6,
                ),
                "average_candidate_reduction": round(
                    statistics.mean(row["avg_candidate_reduction"] for row in agent_rows),
                    6,
                ),
            }
        )
    return sorted(summaries, key=lambda row: row["win_rate"], reverse=True)


def summarize_grouped(rows: list[dict[str, Any]], group_columns: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[column] for column in ["agent", *group_columns])].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        wins = sum(1 for row in group if row["won"])
        item = {"agent": key[0]}
        for column, value in zip(group_columns, key[1:], strict=False):
            item[column] = value
        item.update(
            {
                "games": len(group),
                "win_rate": round(wins / len(group), 6),
                "average_wrong_guesses": round(
                    statistics.mean(row["incorrect_guesses"] for row in group),
                    6,
                ),
            }
        )
        output.append(item)
    return output


def candidate_evolution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in rows:
        counts = json.loads(row["candidate_counts_by_turn"])
        for turn_index, count in enumerate(counts):
            values[(row["agent"], turn_index)].append(int(count))
    return [
        {
            "agent": agent,
            "turn_index": turn_index,
            "average_candidate_count": round(statistics.mean(counts), 6),
            "observations": len(counts),
        }
        for (agent, turn_index), counts in sorted(values.items())
    ]


def paired_bootstrap(rows: list[dict[str, Any]], seed: int, iterations: int = 1000) -> list[dict[str, Any]]:
    by_agent_game = {(row["agent"], row["game_index"]): row["won"] for row in rows}
    agents = sorted({row["agent"] for row in rows})
    game_indices = sorted({row["game_index"] for row in rows})
    rng = random.Random(seed)
    comparisons = []
    for left_index, left in enumerate(agents):
        for right in agents[left_index + 1 :]:
            if any((left, game) not in by_agent_game or (right, game) not in by_agent_game for game in game_indices):
                continue
            observed = statistics.mean(
                int(by_agent_game[(left, game)]) - int(by_agent_game[(right, game)])
                for game in game_indices
            )
            samples = []
            for _ in range(iterations):
                sampled_games = [rng.choice(game_indices) for _ in game_indices]
                samples.append(
                    statistics.mean(
                        int(by_agent_game[(left, game)]) - int(by_agent_game[(right, game)])
                        for game in sampled_games
                    )
                )
            samples.sort()
            comparisons.append(
                {
                    "comparison": f"{left} minus {right}",
                    "win_rate_difference": round(observed, 6),
                    "ci_low": round(samples[int(0.025 * len(samples))], 6),
                    "ci_high": round(samples[int(0.975 * len(samples)) - 1], 6),
                }
            )
    return comparisons


def difficulty_examples(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["game_index"]].append(row)

    output: dict[str, list[dict[str, Any]]] = {
        "decision_tree_wins_entropy_loses": [],
        "entropy_wins_decision_tree_loses": [],
        "candidate_frequency_loses_entropy_wins": [],
        "all_agents_fail": [],
        "all_agents_succeed_one_substantially_fewer_wrong": [],
    }
    for game_index, group in grouped.items():
        by_agent = {row["agent"]: row for row in group}
        word = group[0]["word"]
        base = {
            "game_index": game_index,
            "word": word,
            "word_length": len(word),
            "results": {
                row["agent"]: {
                    "won": bool(row["won"]),
                    "wrong": row["incorrect_guesses"],
                    "turns": row["turns"],
                    "guesses": row["guess_sequence"],
                }
                for row in group
            },
        }
        if by_agent.get("decision_tree", {}).get("won") and not by_agent.get("entropy", {}).get("won", True):
            output["decision_tree_wins_entropy_loses"].append(base)
        if by_agent.get("entropy", {}).get("won") and not by_agent.get("decision_tree", {}).get("won", True):
            output["entropy_wins_decision_tree_loses"].append(base)
        if by_agent.get("entropy", {}).get("won") and not by_agent.get("candidate_frequency", {}).get("won", True):
            output["candidate_frequency_loses_entropy_wins"].append(base)
        if all(not row["won"] for row in group):
            output["all_agents_fail"].append(base)
        if all(row["won"] for row in group):
            wrong_values = [row["incorrect_guesses"] for row in group]
            if max(wrong_values) - min(wrong_values) >= 2:
                output["all_agents_succeed_one_substantially_fewer_wrong"].append(base)
    return {key: value[:25] for key, value in output.items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def output_directory(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return PROJECT_ROOT / "reports" / "results" / args.evaluation_mode


def main() -> None:
    args = parse_args()
    dictionary, words, metadata = resolve_benchmark_words(args)
    if not words:
        raise SystemExit("No benchmark words selected.")

    global_counts: Counter[str] = Counter()
    for word in dictionary:
        global_counts.update(set(word))
    index = CandidateIndex(dictionary)
    agents = build_agents(args, dictionary)
    if not agents:
        raise SystemExit("No agents selected.")

    all_rows = []
    started = time.perf_counter()
    for agent in agents:
        logging.info("Benchmarking %s on %s words [%s]", agent.name, len(words), args.evaluation_mode)
        for game_index, word in enumerate(words):
            all_rows.append(
                play_game(
                    agent,
                    index,
                    dictionary,
                    word,
                    game_index,
                    args.max_lives,
                    args.evaluation_mode,
                    global_counts,
                )
            )
    total_seconds = time.perf_counter() - started
    metadata["agent_count"] = len(agents)
    metadata["agents"] = [agent.name for agent in agents]
    metadata["total_evaluation_time_sec"] = round(total_seconds, 6)

    output_dir = output_directory(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize(all_rows)
    for row in summary_rows:
        row["total_evaluation_time_sec"] = round(total_seconds, 6)

    write_csv(output_dir / "benchmark_games.csv", all_rows)
    write_csv(output_dir / "benchmark_summary.csv", summary_rows)
    write_csv(output_dir / "benchmark_by_length.csv", summarize_grouped(all_rows, ["word_length"]))
    write_csv(
        output_dir / "benchmark_by_word_features.csv",
        summarize_grouped(
            all_rows,
            ["has_repeated_letters", "rare_letter_count", "candidate_size_bucket"],
        ),
    )
    write_csv(output_dir / "candidate_evolution.csv", candidate_evolution(all_rows))
    write_csv(
        output_dir / "paired_comparisons.csv",
        paired_bootstrap(all_rows, seed=args.seed, iterations=args.bootstrap_iterations),
    )
    (output_dir / "difficulty_examples.json").write_text(
        json.dumps(difficulty_examples(all_rows), indent=2),
        encoding="utf-8",
    )
    (output_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    logging.info("Wrote %s benchmark reports to %s", args.evaluation_mode, output_dir)


if __name__ == "__main__":
    main()
