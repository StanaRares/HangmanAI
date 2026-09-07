from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from app.core.candidate_filter import CandidateIndex, letter_pattern
from app.core.dictionary import PROJECT_ROOT, clean_word
from app.core.environment import HangmanEnvironment
from app.core.state import GameState
from app.tree.builder import HangmanTreeBuilder, TreeBuildConfig
from app.tree.model import HangmanDecisionTree, canonical_state_key
from app.tree.serialization import load_tree
from app.tree.vocabulary import (
    load_vocabulary_frame,
    vocabulary_summary,
    weights_for_length,
    words_for_length,
)


@dataclass
class TreeGameSession:
    environment: HangmanEnvironment
    tree: HangmanDecisionTree
    node_id: int
    weighting: str
    model_source: str
    in_vocabulary: bool
    decisions: list[dict[str, Any]] = field(default_factory=list)


class TreeGameService:
    """FastAPI-facing service for the per-length Hangman decision-tree player."""

    def __init__(
        self,
        vocabulary_path: str | Path | None = None,
        models_dir: str | Path | None = None,
        seed: int = 42,
    ) -> None:
        self.vocabulary_path = Path(vocabulary_path) if vocabulary_path else None
        self.models_dir = Path(models_dir) if models_dir else PROJECT_ROOT / "models"
        self.frame = load_vocabulary_frame(self.vocabulary_path)
        self.words = sorted(set(self.frame["word"].astype(str).tolist()))
        weights = {
            str(row.word): 10 ** max(min(float(row.zipf_frequency), 8.0), 0.0)
            for row in self.frame[["word", "zipf_frequency"]].itertuples(index=False)
        }
        self.index = CandidateIndex(self.words, weights=weights)
        self.rng = random.Random(seed)
        self.games: dict[str, TreeGameSession] = {}
        self._tree_cache: dict[tuple[int, str], HangmanDecisionTree] = {}
        self._ephemeral_cache: dict[tuple[int, str], HangmanDecisionTree] = {}

    def health(self) -> dict[str, Any]:
        trained = self.list_trees()
        return {
            "status": "ok",
            "vocabulary_size": len(self.words),
            "available_vocabulary_lengths": sorted(int(length) for length in self.frame["length"].unique()),
            "trained_tree_count": len(trained),
            "available_tree_lengths": sorted({int(row["length"]) for row in trained}),
            "wordfreq_vocabulary_loaded": (PROJECT_ROOT / "data" / "processed" / "wordfreq_vocabulary.parquet").exists(),
        }

    def vocabulary_summary(self) -> list[dict[str, Any]]:
        summary_path = PROJECT_ROOT / "data" / "processed" / "wordfreq_vocabulary_summary.csv"
        if summary_path.exists():
            return self._read_csv(summary_path)
        summary = vocabulary_summary(self.frame)
        return [self._jsonable(row) for row in summary.to_dict(orient="records")]

    def list_trees(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for model_path in sorted(self.models_dir.glob("length_*/*/tree.json.gz")):
            try:
                length = int(model_path.parents[1].name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            weighting = model_path.parent.name
            row = self._tree_files_payload(length, weighting, model_path)
            rows.append(row)
        return sorted(rows, key=lambda row: (int(row["length"]), str(row["weighting"])))

    def tree_summary(self, length: int, weighting: str = "uniform") -> dict[str, Any]:
        tree = self.load_tree(length, weighting)
        payload = self._tree_files_payload(length, weighting, self._tree_path(length, weighting))
        payload["root"] = self.node_payload(length, tree.root_id, weighting)
        return payload

    def node_payload(self, length: int, node_id: int, weighting: str = "uniform") -> dict[str, Any]:
        tree = self.load_tree(length, weighting)
        node = tree.get_node(node_id)
        payload = node.to_dict()
        payload["state_key"] = canonical_state_key(
            length=length,
            pattern=node.pattern,
            guessed_letters=node.guessed_letters,
            incorrect_letters=node.incorrect_letters,
            remaining_lives=node.remaining_lives,
            candidates=node.candidate_sample,
        )
        payload["branches"] = sorted(
            (branch.to_dict() for branch in node.branches.values()),
            key=lambda branch: (-float(branch["probability"]), str(branch["outcome_pattern"])),
        )
        return payload

    def root_analysis(self, length: int, weighting: str = "uniform") -> list[dict[str, Any]]:
        path = PROJECT_ROOT / "reports" / "results" / "root_letter_analysis" / f"length_{length}_{weighting}.csv"
        if not path.exists():
            return []
        return self._read_csv(path)

    def results_by_length(self) -> list[dict[str, Any]]:
        path = PROJECT_ROOT / "reports" / "results" / "performance_by_length.csv"
        if not path.exists():
            rows = []
            for tree in self.list_trees():
                evaluation = self._read_json(
                    self.models_dir / f"length_{tree['length']}" / str(tree["weighting"]) / "evaluation_results.json"
                )
                if evaluation:
                    rows.append(evaluation)
            return rows
        return self._read_csv(path)

    def hardest_words(self, length: int) -> list[dict[str, Any]]:
        path = PROJECT_ROOT / "reports" / "results" / "hardest_words_by_length.csv"
        if not path.exists():
            return []
        return [row for row in self._read_csv(path) if int(row.get("length", -1)) == length]

    def create_game(
        self,
        word: str | None = None,
        length: int | None = None,
        max_lives: int = 6,
        weighting: str = "uniform",
    ) -> tuple[str, GameState]:
        selected = clean_word(word) if word else ""
        if word and not selected:
            raise ValueError("Provided word must contain only letters a-z.")
        if selected:
            length = len(selected)
        if length is None:
            length = self._default_play_length(weighting)
        tree, source = self.load_tree_for_play(length, weighting)
        bucket = words_for_length(self.frame, length)
        if not bucket:
            raise ValueError(f"No vocabulary bucket is available for {length}-letter words.")
        selected_word = selected or self.rng.choice(bucket)
        environment_words = sorted(set(bucket + [selected_word]))
        environment = HangmanEnvironment(environment_words, max_lives=max_lives, seed=self.rng.randint(0, 10**9))
        state = environment.reset(selected_word)
        game_id = str(uuid4())
        self.games[game_id] = TreeGameSession(
            environment=environment,
            tree=tree,
            node_id=tree.root_id,
            weighting=weighting,
            model_source=source,
            in_vocabulary=selected_word in set(bucket),
        )
        return game_id, state

    def get_game(self, game_id: str) -> TreeGameSession:
        if game_id not in self.games:
            raise KeyError(f"Unknown game_id: {game_id}")
        return self.games[game_id]

    def manual_guess(self, game_id: str, letter: str) -> tuple[GameState, dict[str, Any]]:
        session = self.get_game(game_id)
        previous = session.node_id
        guess = clean_word(letter)
        if len(guess) != 1:
            raise ValueError("Guess must be a single letter a-z.")
        outcome = letter_pattern(session.environment.solution, guess)
        state = session.environment.step(guess)
        next_node_id = session.tree.next_node_id(previous, outcome)
        if next_node_id is not None:
            session.node_id = next_node_id
        decision = self._decision_payload(session, previous, guess, outcome, next_node_id)
        session.decisions.append(decision)
        return state, decision

    def tree_step(self, game_id: str) -> tuple[GameState, dict[str, Any]]:
        session = self.get_game(game_id)
        state = session.environment.get_state()
        if state.is_finished:
            raise ValueError("Cannot guess after the game has finished.")
        previous = session.node_id
        guess = session.tree.next_guess(previous, state)
        if guess is None:
            raise ValueError("Tree has no available guess for this state.")
        outcome = letter_pattern(session.environment.solution, guess)
        state = session.environment.step(guess)
        next_node_id = session.tree.next_node_id(previous, outcome)
        if next_node_id is not None:
            session.node_id = next_node_id
        decision = self._decision_payload(session, previous, guess, outcome, next_node_id)
        session.decisions.append(decision)
        return state, decision

    def tree_play(self, game_id: str, max_steps: int = 26) -> tuple[GameState, list[dict[str, Any]]]:
        session = self.get_game(game_id)
        decisions: list[dict[str, Any]] = []
        for _ in range(max_steps):
            if session.environment.get_state().is_finished:
                break
            state, decision = self.tree_step(game_id)
            decisions.append(decision)
            if state.is_finished:
                break
        return session.environment.get_state(), decisions

    def candidate_analysis(self, state: GameState) -> dict[str, Any]:
        candidates = self.index.candidates(state)
        return {
            "candidate_count": len(candidates),
            "candidate_sample": list(candidates[:50]),
        }

    def load_tree_for_play(self, length: int, weighting: str = "uniform") -> tuple[HangmanDecisionTree, str]:
        path = self._tree_path(length, weighting)
        if path.exists():
            return self.load_tree(length, weighting), "serialized"
        key = (length, weighting)
        if key in self._ephemeral_cache:
            return self._ephemeral_cache[key], "ephemeral-demo"
        words = words_for_length(self.frame, length)
        if len(words) > 500:
            raise ValueError(
                f"No serialized {length}-letter tree exists. Train it first with "
                f"`python scripts/train_tree.py --length {length}`."
            )
        weights = weights_for_length(self.frame, length) if weighting == "wordfreq" else None
        config = TreeBuildConfig.from_profile(
            length=length,
            profile="fast",
            weighting=weighting,  # type: ignore[arg-type]
            node_budget=5000,
        )
        tree = HangmanTreeBuilder(words, weights, config).build()
        tree.metadata["model_source"] = "ephemeral-demo"
        self._ephemeral_cache[key] = tree
        return tree, "ephemeral-demo"

    def load_tree(self, length: int, weighting: str = "uniform") -> HangmanDecisionTree:
        key = (length, weighting)
        if key not in self._tree_cache:
            path = self._tree_path(length, weighting)
            if not path.exists():
                raise FileNotFoundError(f"No serialized {length}-letter {weighting} tree found at {path}.")
            self._tree_cache[key] = load_tree(path)
        return self._tree_cache[key]

    def _default_play_length(self, weighting: str) -> int:
        trained = [row["length"] for row in self.list_trees() if row["weighting"] == weighting]
        if trained:
            return int(sorted(trained)[0])
        counts = self.frame.groupby("length")["word"].count().sort_index()
        demo_lengths = [int(length) for length, count in counts.items() if int(count) <= 500]
        if demo_lengths:
            return demo_lengths[0]
        return int(counts.index[0])

    def _tree_path(self, length: int, weighting: str) -> Path:
        return self.models_dir / f"length_{length}" / weighting / "tree.json.gz"

    def _tree_files_payload(self, length: int, weighting: str, model_path: Path) -> dict[str, Any]:
        model_dir = model_path.parent
        metadata = self._read_json(model_dir / "metadata.json")
        evaluation = self._read_json(model_dir / "evaluation_results.json")
        training = self._read_json(model_dir / "training_statistics.json")
        vocabulary = self._read_json(model_dir / "vocabulary_stats.json")
        payload: dict[str, Any] = {
            "length": length,
            "weighting": weighting,
            "model_path": str(model_path),
            "model_file_size": model_path.stat().st_size if model_path.exists() else 0,
            "metadata": metadata,
            "evaluation": evaluation,
            "training_statistics": training,
            "vocabulary": vocabulary,
            "best_first_guess": training.get("best_first_guess") or evaluation.get("best_first_guess"),
            "node_count": training.get("node_count") or evaluation.get("node_count"),
            "leaf_count": training.get("leaf_count") or evaluation.get("leaf_count"),
            "max_depth": training.get("max_depth") or evaluation.get("maximum_tree_depth"),
            "win_rate": evaluation.get("win_rate") or training.get("win_rate"),
            "average_mistakes": evaluation.get("average_wrong_guesses") or training.get("average_wrong_guesses"),
            "training_strategy": training.get("strategy") or evaluation.get("training_strategy") or metadata.get("requested_strategy"),
            "training_time": training.get("training_time") or evaluation.get("training_time") or metadata.get("training_seconds"),
            "words": vocabulary.get("words") or evaluation.get("total_words") or metadata.get("word_count"),
        }
        return payload

    def _decision_payload(
        self,
        session: TreeGameSession,
        previous_node_id: int,
        guess: str,
        outcome: str,
        next_node_id: int | None,
    ) -> dict[str, Any]:
        node = session.tree.get_node(previous_node_id)
        branch = node.branches.get(outcome)
        return {
            "tree_length": session.tree.length,
            "weighting": session.weighting,
            "model_source": session.model_source,
            "node_id": previous_node_id,
            "next_node_id": next_node_id,
            "branch_found": next_node_id is not None,
            "guess": guess,
            "outcome_pattern": outcome,
            "resulting_pattern": session.environment.get_state().pattern,
            "node": {
                "candidate_count": node.candidate_count,
                "win_probability": node.win_probability,
                "average_mistakes": node.average_mistakes,
                "depth": node.depth,
                "branch_count": len(node.children),
                "fallback_letters": list(node.fallback_letters[:8]),
            },
            "branch": branch.to_dict() if branch else {"outcome_pattern": outcome, "node_id": None},
        }

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @classmethod
    def _read_csv(cls, path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [cls._jsonable(row) for row in csv.DictReader(handle)]

    @staticmethod
    def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in row.items():
            if pd.isna(value):
                output[key] = None
            elif isinstance(value, str):
                stripped = value.strip()
                if stripped == "":
                    output[key] = ""
                    continue
                try:
                    number = float(stripped)
                except ValueError:
                    output[key] = value
                else:
                    output[key] = int(number) if number.is_integer() else number
            elif hasattr(value, "item"):
                output[key] = value.item()
            else:
                output[key] = value
        return output
