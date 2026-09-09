from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.candidate_filter import letter_pattern
from app.core.environment import HangmanEnvironment
from app.tree.model import HangmanDecisionTree


@dataclass(frozen=True)
class TreeGameResult:
    word: str
    won: bool
    status: str
    incorrect_guesses: int
    total_guesses: int
    remaining_lives: int
    traversal_depth: int
    guess_sequence: tuple[str, ...]
    node_sequence: tuple[int, ...]
    candidate_counts: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "won": self.won,
            "status": self.status,
            "incorrect_guesses": self.incorrect_guesses,
            "total_guesses": self.total_guesses,
            "remaining_lives": self.remaining_lives,
            "traversal_depth": self.traversal_depth,
            "guess_sequence": " ".join(self.guess_sequence),
            "node_sequence": " ".join(str(node_id) for node_id in self.node_sequence),
            "candidate_counts": " ".join(str(count) for count in self.candidate_counts),
        }


def evaluate_tree_on_word(
    tree: HangmanDecisionTree,
    word: str,
    max_lives: int | None = None,
) -> TreeGameResult:
    environment = HangmanEnvironment([word], max_lives=max_lives or tree.objective.max_lives)
    state = environment.reset(word)
    node_id = tree.root_id
    node_sequence = [node_id]
    guesses: list[str] = []
    candidate_counts = [tree.get_node(node_id).candidate_count]

    while not state.is_finished:
        guess = tree.next_guess(node_id, state)
        if guess is None:
            break
        if guess in state.guessed_letters:
            raise ValueError(f"Tree node {node_id} repeated already-guessed letter '{guess}'.")
        guesses.append(guess)
        previous_node_id = node_id
        state = environment.step(guess)
        outcome = letter_pattern(word, guess)
        next_node_id = tree.next_node_id(previous_node_id, outcome)
        if next_node_id is None:
            raise ValueError(
                f"Tree node {previous_node_id} has no explicit child for outcome {outcome} while evaluating {word}."
            )
        node_id = next_node_id
        node_sequence.append(node_id)
        candidate_counts.append(tree.get_node(node_id).candidate_count)

    return TreeGameResult(
        word=word,
        won=state.status == "won",
        status=state.status,
        incorrect_guesses=len(state.incorrect_letters),
        total_guesses=len(state.guessed_letters),
        remaining_lives=state.remaining_lives,
        traversal_depth=len(node_sequence) - 1,
        guess_sequence=tuple(guesses),
        node_sequence=tuple(node_sequence),
        candidate_counts=tuple(candidate_counts),
    )


def evaluate_tree(
    tree: HangmanDecisionTree,
    words: list[str],
    model_path: str | Path | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    results = [evaluate_tree_on_word(tree, word) for word in words]
    elapsed = time.perf_counter() - started
    wins = sum(result.won for result in results)
    wrong = [result.incorrect_guesses for result in results]
    total_guesses = [result.total_guesses for result in results]
    remaining = [result.remaining_lives for result in results]
    traversal = [result.traversal_depth for result in results]
    row_weights = [max(float(weights.get(result.word, 1.0)), 0.0) if weights else 1.0 for result in results]
    total_weight = sum(row_weights) or 1.0

    def weighted_mean(values: list[int | float]) -> float:
        return sum(float(value) * weight for value, weight in zip(values, row_weights)) / total_weight

    model_size = Path(model_path).stat().st_size if model_path and Path(model_path).exists() else 0
    summary = {
        "length": tree.length,
        "total_words": len(words),
        "wins": wins,
        "losses": len(words) - wins,
        "win_rate": wins / len(words) if words else 0.0,
        "average_wrong_guesses": statistics.mean(wrong) if wrong else 0.0,
        "median_wrong_guesses": statistics.median(wrong) if wrong else 0.0,
        "average_total_guesses": statistics.mean(total_guesses) if total_guesses else 0.0,
        "average_remaining_lives": statistics.mean(remaining) if remaining else 0.0,
        "maximum_tree_depth": tree.max_depth,
        "average_traversal_depth": statistics.mean(traversal) if traversal else 0.0,
        "node_count": tree.node_count,
        "leaf_count": tree.leaf_count,
        "model_file_size": model_size,
        "training_time": tree.metadata.get("training_seconds", 0.0),
        "evaluation_time": elapsed,
        "best_first_guess": tree.best_first_guess or "",
        "training_strategy": tree.strategy,
        "weighting": tree.objective.weighting,
        "evaluation_weighting": "wordfreq" if weights else "uniform",
        "weighted_win_rate": weighted_mean([float(result.won) for result in results]),
        "weighted_average_wrong_guesses": weighted_mean(wrong),
        "weighted_average_total_guesses": weighted_mean(total_guesses),
        "weighted_average_remaining_lives": weighted_mean(remaining),
    }
    return summary, [result.to_dict() for result in results]
