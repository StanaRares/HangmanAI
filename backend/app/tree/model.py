from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from app.core.candidate_filter import letter_pattern
from app.core.state import ALPHABET, UNKNOWN, GameState

WeightingMode = Literal["uniform", "wordfreq"]
TreeStrategy = Literal["greedy", "bounded-lookahead", "exact", "near-exact"]
LeafType = Literal["solved", "loss", "deterministic", "budget", "depth_limited", "exhausted"]


@dataclass(frozen=True)
class TreeObjective:
    """Objective used to rank candidate policy trees."""

    max_lives: int = 6
    weighting: WeightingMode = "uniform"
    primary: str = "maximize_win_rate"
    tie_breakers: tuple[str, ...] = (
        "lowest_average_wrong_guesses",
        "lowest_average_total_guesses",
        "lowest_expected_depth",
        "smallest_tree",
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tie_breakers"] = list(self.tie_breakers)
        return payload


@dataclass(frozen=True)
class TreeStats:
    win_probability: float
    average_mistakes: float
    average_total_guesses: float
    average_remaining_lives: float
    expected_depth: float
    node_count: int
    leaf_count: int
    max_depth: int

    def score_tuple(self) -> tuple[float, float, float, float, float]:
        return (
            self.win_probability,
            -self.average_mistakes,
            -self.average_total_guesses,
            -self.expected_depth,
            -self.node_count,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BranchInfo:
    outcome_pattern: str
    node_id: int
    word_count: int
    probability: float
    is_miss: bool
    resulting_pattern: str
    remaining_lives: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TreeNode:
    """A multiway Hangman decision-tree node.

    Internal nodes prescribe exactly one letter guess. Child branches are keyed
    by the full position-pattern outcome of that guess, such as ``00100``.
    """

    node_id: int
    depth: int
    pattern: str
    guessed_letters: tuple[str, ...]
    incorrect_letters: tuple[str, ...]
    remaining_lives: int
    candidate_count: int
    guess: str | None = None
    leaf_type: LeafType | None = None
    children: dict[str, int] = field(default_factory=dict)
    branches: dict[str, BranchInfo] = field(default_factory=dict)
    win_probability: float = 0.0
    average_mistakes: float = 0.0
    average_total_guesses: float = 0.0
    average_remaining_lives: float = 0.0
    expected_depth: float = 0.0
    candidate_sample: tuple[str, ...] = field(default_factory=tuple)
    fallback_letters: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_leaf(self) -> bool:
        return self.guess is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "pattern": self.pattern,
            "guessed_letters": list(self.guessed_letters),
            "incorrect_letters": list(self.incorrect_letters),
            "remaining_lives": self.remaining_lives,
            "candidate_count": self.candidate_count,
            "guess": self.guess,
            "leaf_type": self.leaf_type,
            "children": self.children,
            "branches": {key: value.to_dict() for key, value in self.branches.items()},
            "win_probability": self.win_probability,
            "average_mistakes": self.average_mistakes,
            "average_total_guesses": self.average_total_guesses,
            "average_remaining_lives": self.average_remaining_lives,
            "expected_depth": self.expected_depth,
            "candidate_sample": list(self.candidate_sample),
            "fallback_letters": list(self.fallback_letters),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TreeNode":
        return cls(
            node_id=int(payload["node_id"]),
            depth=int(payload["depth"]),
            pattern=str(payload["pattern"]),
            guessed_letters=tuple(payload.get("guessed_letters", [])),
            incorrect_letters=tuple(payload.get("incorrect_letters", [])),
            remaining_lives=int(payload["remaining_lives"]),
            candidate_count=int(payload["candidate_count"]),
            guess=payload.get("guess"),
            leaf_type=payload.get("leaf_type"),
            children={str(key): int(value) for key, value in payload.get("children", {}).items()},
            branches={
                str(key): BranchInfo(
                    outcome_pattern=str(value["outcome_pattern"]),
                    node_id=int(value["node_id"]),
                    word_count=int(value["word_count"]),
                    probability=float(value["probability"]),
                    is_miss=bool(value["is_miss"]),
                    resulting_pattern=str(value["resulting_pattern"]),
                    remaining_lives=int(value["remaining_lives"]),
                )
                for key, value in payload.get("branches", {}).items()
            },
            win_probability=float(payload.get("win_probability", 0.0)),
            average_mistakes=float(payload.get("average_mistakes", 0.0)),
            average_total_guesses=float(payload.get("average_total_guesses", 0.0)),
            average_remaining_lives=float(payload.get("average_remaining_lives", 0.0)),
            expected_depth=float(payload.get("expected_depth", 0.0)),
            candidate_sample=tuple(payload.get("candidate_sample", [])),
            fallback_letters=tuple(payload.get("fallback_letters", [])),
        )


@dataclass
class HangmanDecisionTree:
    length: int
    objective: TreeObjective
    strategy: TreeStrategy
    root_id: int
    nodes: dict[int, TreeNode]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def root(self) -> TreeNode:
        return self.nodes[self.root_id]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def leaf_count(self) -> int:
        return sum(1 for node in self.nodes.values() if node.is_leaf)

    @property
    def max_depth(self) -> int:
        return max((node.depth for node in self.nodes.values()), default=0)

    @property
    def best_first_guess(self) -> str | None:
        return self.root.guess

    def get_node(self, node_id: int) -> TreeNode:
        if node_id not in self.nodes:
            raise KeyError(f"Unknown tree node: {node_id}")
        return self.nodes[node_id]

    def next_node_id(self, node_id: int, outcome_pattern: str) -> int | None:
        return self.get_node(node_id).children.get(outcome_pattern)

    def next_guess(self, node_id: int, state: GameState) -> str | None:
        node = self.get_node(node_id)
        if node.guess and node.guess not in state.guessed_letters:
            return node.guess
        for letter in node.fallback_letters:
            if letter not in state.guessed_letters:
                return letter
        return next((letter for letter in ALPHABET if letter not in state.guessed_letters), None)

    def branch_payload(self, node_id: int, outcome_pattern: str) -> dict[str, Any]:
        node = self.get_node(node_id)
        branch = node.branches.get(outcome_pattern)
        return branch.to_dict() if branch else {"outcome_pattern": outcome_pattern, "node_id": None}

    def to_dict(self) -> dict[str, Any]:
        return {
            "length": self.length,
            "objective": self.objective.to_dict(),
            "strategy": self.strategy,
            "root_id": self.root_id,
            "nodes": {str(node_id): node.to_dict() for node_id, node in self.nodes.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HangmanDecisionTree":
        objective_payload = payload.get("objective", {})
        objective = TreeObjective(
            max_lives=int(objective_payload.get("max_lives", 6)),
            weighting=objective_payload.get("weighting", "uniform"),
            primary=objective_payload.get("primary", "maximize_win_rate"),
            tie_breakers=tuple(objective_payload.get("tie_breakers", TreeObjective().tie_breakers)),
        )
        return cls(
            length=int(payload["length"]),
            objective=objective,
            strategy=payload.get("strategy", "greedy"),
            root_id=int(payload.get("root_id", 0)),
            nodes={
                int(node_id): TreeNode.from_dict(node_payload)
                for node_id, node_payload in payload.get("nodes", {}).items()
            },
            metadata=payload.get("metadata", {}),
        )


def merge_pattern(pattern: str, word: str, letter: str) -> str:
    chars = list(pattern)
    for index, char in enumerate(word):
        if char == letter:
            chars[index] = letter
    return "".join(chars)


def apply_outcome(pattern: str, letter: str, outcome_pattern: str) -> str:
    chars = list(pattern)
    for index, flag in enumerate(outcome_pattern):
        if flag == "1":
            chars[index] = letter
    return "".join(chars)


def outcome_for_word(word: str, letter: str) -> str:
    return letter_pattern(word, letter)


def is_solved_pattern(pattern: str) -> bool:
    return UNKNOWN not in pattern


def canonical_state_key(
    *,
    length: int,
    pattern: str,
    guessed_letters: Iterable[str],
    incorrect_letters: Iterable[str],
    remaining_lives: int,
    candidates: Iterable[str | int],
) -> str:
    """Return a stable compact identity for equivalent Hangman tree states."""
    payload = "|".join(
        [
            str(length),
            pattern,
            "".join(sorted(guessed_letters)),
            "".join(sorted(incorrect_letters)),
            str(remaining_lives),
            ",".join(sorted(str(candidate) for candidate in candidates)),
        ]
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()
