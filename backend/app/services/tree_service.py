from __future__ import annotations

import csv
import json
import os
import random
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from app.core.candidate_filter import CandidateIndex, letter_pattern
from app.core.dictionary import PROJECT_ROOT, clean_word
from app.core.environment import HangmanEnvironment
from app.core.state import GameState
from app.tree.builder import HangmanTreeBuilder, TreeBuildConfig
from app.tree.model import TREE_FORMAT_VERSION, HangmanDecisionTree, TreeNode, canonical_state_key
from app.tree.serialization import load_tree, save_tree_atomic
from app.tree.vocabulary import (
    load_vocabulary_frame,
    vocabulary_summary,
    weights_for_length,
    words_for_length,
)


class ExpansionInProgressError(RuntimeError):
    """Raised when a tree node expansion is already running."""


class TreeConsistencyError(RuntimeError):
    """Raised when the stored tree cannot represent a valid traversal step."""


@dataclass
class TreeGameSession:
    environment: HangmanEnvironment
    tree: HangmanDecisionTree
    node_id: int | None
    weighting: str
    model_source: str
    in_vocabulary: bool
    tree_position_valid: bool = True
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
        self._expansion_locks: dict[tuple[int, str, int], threading.Lock] = {}
        self._expansion_locks_guard = threading.Lock()

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
        reconstructed_candidates = self.reconstruct_node_candidates(length, node)
        payload = node.to_dict()
        payload["state_key"] = canonical_state_key(
            length=length,
            pattern=node.pattern,
            guessed_letters=node.guessed_letters,
            incorrect_letters=node.incorrect_letters,
            remaining_lives=node.remaining_lives,
            candidates=reconstructed_candidates,
        )
        payload["reconstructed_candidate_count"] = len(reconstructed_candidates)
        payload["candidate_count_mismatch"] = len(reconstructed_candidates) != node.candidate_count
        payload["expandable"] = self._is_expandable_node(node)
        payload["is_terminal"] = node.is_terminal
        payload["is_checkpoint"] = node.is_checkpoint
        payload["needs_expansion"] = node.needs_expansion
        payload["extension_loaded"] = self._extension_path(length, weighting, node_id).exists() and node.guess is not None
        payload["branches"] = sorted(
            (branch.to_dict() for branch in node.branches.values()),
            key=lambda branch: (-float(branch["probability"]), str(branch["outcome_pattern"])),
        )
        return payload

    def materialized_node_payload(
        self,
        length: int,
        node_id: int,
        weighting: str = "uniform",
        node_budget: int = 20000,
    ) -> dict[str, Any]:
        tree = self.load_tree(length, weighting)
        node = tree.get_node(node_id)
        if self._is_expandable_node(node):
            self.expand_node(length, node_id, weighting=weighting, node_budget=node_budget)
        return self.node_payload(length, node_id, weighting)

    def expand_node(
        self,
        length: int,
        node_id: int,
        weighting: str = "uniform",
        node_budget: int = 20000,
        profile: str = "fast",
    ) -> dict[str, Any]:
        lock = self._expansion_lock(length, weighting, node_id)
        if not lock.acquire(blocking=False):
            raise ExpansionInProgressError(f"Expansion for node {node_id} is already in progress.")
        file_lock_path: Path | None = None
        try:
            file_lock_path = self._acquire_expansion_file_lock(length, weighting, node_id)
            if file_lock_path is None:
                raise ExpansionInProgressError(f"Expansion for node {node_id} is already in progress.")
            tree = self.load_tree(length, weighting)
            extension_path = self._extension_path(length, weighting, node_id)
            node = tree.get_node(node_id)
            if node.guess is None and extension_path.exists():
                self._load_extensions(tree, length, weighting)
                node = tree.get_node(node_id)
            if node.guess is not None:
                payload = self.node_payload(length, node_id, weighting)
                payload["expanded"] = False
                payload["already_expanded"] = True
                return payload
            if not self._is_expandable_node(node):
                raise ValueError(f"Node {node_id} is not a budget/depth-limited expandable checkpoint.")

            reconstructed_candidates = self.reconstruct_node_candidates(length, node)
            if len(reconstructed_candidates) != node.candidate_count:
                raise ValueError(
                    "Reconstructed candidate count "
                    f"{len(reconstructed_candidates)} does not match stored node count {node.candidate_count}."
                )
            if not reconstructed_candidates:
                raise ValueError(f"Node {node_id} has no reconstructable candidates to expand.")

            try:
                config = TreeBuildConfig.from_profile(
                    length=length,
                    profile=profile,  # type: ignore[arg-type]
                    max_lives=tree.objective.max_lives,
                    weighting=weighting,  # type: ignore[arg-type]
                    node_budget=node_budget,
                )
            except KeyError as exc:
                raise ValueError(f"Unknown expansion profile: {profile}") from exc
            config = replace(config, max_depth=node.depth + config.max_depth)
            weights = weights_for_length(self.frame, length) if weighting == "wordfreq" else None
            reserved_node_ids = set(tree.nodes)
            reserved_node_ids.discard(node_id)
            builder = HangmanTreeBuilder(
                words_for_length(self.frame, length),
                weights=weights,
                config=config,
                start_node_id=node_id,
                reserved_node_ids=reserved_node_ids,
            )
            extension = builder.build_from_state(
                candidates=reconstructed_candidates,
                pattern=node.pattern,
                guessed_letters=node.guessed_letters,
                incorrect_letters=node.incorrect_letters,
                remaining_lives=node.remaining_lives,
                depth=node.depth,
                root_node_id=node_id,
            )
            root = extension.get_node(extension.root_id)
            if root.guess is None:
                raise ValueError(
                    f"Expansion budget {node_budget} was not large enough to create a decision node."
                )
            extension.metadata["extends_node_id"] = node_id
            extension.metadata["base_tree_length"] = length
            extension.metadata["base_tree_weighting"] = weighting
            extension.metadata["base_tree_format_version"] = tree.metadata.get("tree_format_version")
            extension.metadata["reconstructed_candidate_count"] = len(reconstructed_candidates)
            extension.metadata["tree_format_version"] = TREE_FORMAT_VERSION

            save_tree_atomic(extension, extension_path)
            self._merge_extension(tree, extension, extension_path)
            payload = self.node_payload(length, node_id, weighting)
            payload["expanded"] = True
            payload["already_expanded"] = False
            payload["extension_path"] = str(extension_path)
            payload["extension_node_count"] = extension.node_count
            return payload
        finally:
            if file_lock_path is not None:
                try:
                    file_lock_path.unlink()
                except FileNotFoundError:
                    pass
            lock.release()

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
        if selected and selected_word not in set(bucket):
            raise ValueError(
                "Strict decision-tree play requires a hidden word from the trained vocabulary for that length."
            )
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

    def manual_guess(self, game_id: str, letter: str) -> tuple[GameState, dict[str, Any] | None]:
        session = self.get_game(game_id)
        guess = clean_word(letter)
        if len(guess) != 1:
            raise ValueError("Guess must be a single letter a-z.")

        state = session.environment.get_state()
        if state.is_finished:
            raise ValueError("Cannot guess after the game has finished.")

        if session.tree_position_valid and session.node_id is not None:
            node = self.ensure_decision_node(session)
            if node.guess == guess:
                return self._advance_tree_decision(session, node, state)

        next_state = session.environment.step(guess)
        session.node_id = None
        session.tree_position_valid = False
        return next_state, None

    def tree_step(self, game_id: str) -> tuple[GameState, dict[str, Any]]:
        session = self.get_game(game_id)
        state = session.environment.get_state()
        if state.is_finished:
            raise ValueError("Cannot guess after the game has finished.")
        if not session.in_vocabulary:
            raise ValueError("Strict decision-tree traversal requires an in-vocabulary hidden word.")
        if not session.tree_position_valid or session.node_id is None:
            raise ValueError(
                "Manual off-policy guesses have left strict tree traversal. Start a new word or let the tree play."
            )

        node = self.ensure_decision_node(session)
        return self._advance_tree_decision(session, node, state)

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
        tree = self._tree_cache[key]
        self._load_extensions(tree, length, weighting)
        return tree

    def reconstruct_node_candidates(self, length: int, node: TreeNode) -> tuple[str, ...]:
        pattern_letters = frozenset(letter for letter in node.pattern if letter != "_")
        guessed_letters = frozenset(node.guessed_letters) | frozenset(node.incorrect_letters) | pattern_letters
        state = GameState(
            word_length=length,
            pattern=node.pattern,
            guessed_letters=guessed_letters,
            incorrect_letters=frozenset(node.incorrect_letters),
            remaining_lives=node.remaining_lives,
            max_lives=max(node.remaining_lives, 1),
        )
        return self.index.candidates(state)

    def ensure_decision_node(
        self,
        session: TreeGameSession,
        expansion_budget: int = 20000,
    ) -> TreeNode:
        if session.node_id is None:
            raise TreeConsistencyError("Strict tree traversal has no current node.")

        tree = self.load_tree(session.tree.length, session.weighting)
        session.tree = tree
        node = tree.get_node(session.node_id)
        if node.guess is not None:
            return node
        if node.is_terminal:
            return node
        if self._is_expandable_node(node):
            self.expand_node(
                length=session.tree.length,
                node_id=node.node_id,
                weighting=session.weighting,
                node_budget=expansion_budget,
            )
            tree = self.load_tree(session.tree.length, session.weighting)
            session.tree = tree
            node = tree.get_node(session.node_id)
            if node.guess is not None or node.is_terminal:
                return node
        raise TreeConsistencyError(f"Node {session.node_id} is neither a decision nor a terminal node.")

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

    def _extension_dir(self, length: int, weighting: str) -> Path:
        return self.models_dir / f"length_{length}" / weighting / "extensions"

    def _extension_path(self, length: int, weighting: str, node_id: int) -> Path:
        return self._extension_dir(length, weighting) / f"node_{node_id}.json.gz"

    def _is_expandable_node(self, node: TreeNode) -> bool:
        return node.needs_expansion and node.remaining_lives > 0

    def _expansion_lock(self, length: int, weighting: str, node_id: int) -> threading.Lock:
        key = (length, weighting, node_id)
        with self._expansion_locks_guard:
            if key not in self._expansion_locks:
                self._expansion_locks[key] = threading.Lock()
            return self._expansion_locks[key]

    def _acquire_expansion_file_lock(self, length: int, weighting: str, node_id: int) -> Path | None:
        extension_path = self._extension_path(length, weighting, node_id)
        lock_path = extension_path.with_name(f"{extension_path.name}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return None
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        return lock_path

    def _load_extensions(self, tree: HangmanDecisionTree, length: int, weighting: str) -> None:
        extension_dir = self._extension_dir(length, weighting)
        if not extension_dir.exists():
            return
        loaded = set(tree.metadata.setdefault("loaded_extensions", []))
        for path in sorted(extension_dir.glob("node_*.json.gz")):
            path_key = str(path)
            if path_key in loaded:
                continue
            extension = load_tree(path)
            self._merge_extension(tree, extension, path)
            loaded.add(path_key)
        tree.metadata["loaded_extensions"] = sorted(loaded)

    def _merge_extension(self, tree: HangmanDecisionTree, extension: HangmanDecisionTree, path: Path) -> None:
        if extension.metadata.get("tree_format_version") != TREE_FORMAT_VERSION:
            raise ValueError(
                f"Extension {path} has incompatible tree_format_version "
                f"{extension.metadata.get('tree_format_version')!r}; expected {TREE_FORMAT_VERSION}."
            )
        base_format_version = tree.metadata.get("tree_format_version")
        if extension.metadata.get("base_tree_format_version") != base_format_version:
            raise ValueError(
                f"Extension {path} was built against base tree format "
                f"{extension.metadata.get('base_tree_format_version')!r}, "
                f"but the loaded base tree is {base_format_version!r}."
            )
        root = extension.get_node(extension.root_id)
        if root.node_id not in tree.nodes:
            raise KeyError(f"Extension root node {root.node_id} does not exist in the base tree.")
        existing = tree.nodes[root.node_id]
        if (
            existing.pattern != root.pattern
            or tuple(existing.guessed_letters) != tuple(root.guessed_letters)
            or tuple(existing.incorrect_letters) != tuple(root.incorrect_letters)
            or existing.remaining_lives != root.remaining_lives
        ):
            raise ValueError(f"Extension {path} does not match the base node state.")
        for extension_node_id, extension_node in extension.nodes.items():
            if extension_node_id in tree.nodes and extension_node_id != root.node_id:
                existing_payload = tree.nodes[extension_node_id].to_dict()
                if existing_payload != extension_node.to_dict():
                    raise ValueError(f"Extension {path} conflicts with existing node {extension_node_id}.")
        tree.nodes.update(extension.nodes)
        extensions = dict(tree.metadata.setdefault("extensions", {}))
        extensions[str(root.node_id)] = {
            "path": str(path),
            "node_count": extension.node_count,
            "training_seconds": extension.metadata.get("training_seconds"),
            "node_budget": extension.metadata.get("node_budget"),
        }
        tree.metadata["extensions"] = extensions

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
        next_node_id: int,
    ) -> dict[str, Any]:
        node = session.tree.get_node(previous_node_id)
        branch = node.branches.get(outcome)
        return {
            "tree_length": session.tree.length,
            "weighting": session.weighting,
            "model_source": session.model_source,
            "node_id": previous_node_id,
            "next_node_id": next_node_id,
            "branch_found": True,
            "guess": guess,
            "outcome_pattern": outcome,
            "resulting_pattern": session.environment.get_state().pattern,
            "path_segment": f"{previous_node_id} -{guess.upper()}/{outcome}-> {next_node_id}",
            "node": {
                "candidate_count": node.candidate_count,
                "win_probability": node.win_probability,
                "average_mistakes": node.average_mistakes,
                "depth": node.depth,
                "branch_count": len(node.children),
            },
            "branch": branch.to_dict() if branch else {"outcome_pattern": outcome, "node_id": next_node_id},
        }

    def _advance_tree_decision(
        self,
        session: TreeGameSession,
        node: TreeNode,
        state: GameState,
    ) -> tuple[GameState, dict[str, Any]]:
        if node.guess is None:
            if node.is_terminal:
                raise ValueError("Tree traversal reached a terminal node before the game finished.")
            raise TreeConsistencyError(f"Node {node.node_id} has no explicit decision.")
        if node.guess in state.guessed_letters:
            raise TreeConsistencyError(
                f"Node {node.node_id} guesses '{node.guess}', but that letter is already in the game state."
            )

        previous = node.node_id
        outcome = letter_pattern(session.environment.solution, node.guess)
        next_state = session.environment.step(node.guess)
        next_node_id = self._require_child_node(session, previous, outcome)
        session.node_id = next_node_id
        if not next_state.is_finished:
            self.ensure_decision_node(session)
        decision = self._decision_payload(session, previous, node.guess, outcome, next_node_id)
        session.decisions.append(decision)
        return next_state, decision

    def _require_child_node(self, session: TreeGameSession, node_id: int, outcome: str) -> int:
        next_node_id = session.tree.next_node_id(node_id, outcome)
        if next_node_id is not None:
            return next_node_id

        self._load_extensions(session.tree, session.tree.length, session.weighting)
        next_node_id = session.tree.next_node_id(node_id, outcome)
        if next_node_id is not None:
            return next_node_id
        raise TreeConsistencyError(
            f"Tree node {node_id} has no explicit child branch for observed outcome {outcome}."
        )

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
