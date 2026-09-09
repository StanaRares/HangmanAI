from __future__ import annotations

import hashlib
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

from app.core.state import ALPHABET, UNKNOWN
from app.tree.model import (
    BranchInfo,
    HangmanDecisionTree,
    TREE_FORMAT_VERSION,
    TreeNode,
    TreeObjective,
    TreeStats,
    WeightingMode,
    apply_outcome,
    is_solved_pattern,
    outcome_for_word,
)

TrainingProfile = Literal["fast", "balanced", "deep", "exact"]


@dataclass(frozen=True)
class TreeBuildConfig:
    length: int
    max_lives: int = 6
    weighting: WeightingMode = "uniform"
    strategy: str = "greedy"
    lookahead: int = 1
    max_depth: int = 32
    node_budget: int = 25000
    lookahead_candidate_limit: int = 250
    exact_candidate_limit: int = 60
    profile: TrainingProfile = "balanced"
    sample_size: int = 12
    pruning_min_gain: float = 0.0

    def __post_init__(self) -> None:
        if self.node_budget < 1:
            raise ValueError("Tree node_budget must allow at least one node.")

    @classmethod
    def from_profile(
        cls,
        length: int,
        profile: TrainingProfile,
        max_lives: int = 6,
        weighting: WeightingMode = "uniform",
        lookahead: int | None = None,
        node_budget: int | None = None,
    ) -> "TreeBuildConfig":
        defaults = {
            "fast": dict(strategy="greedy", lookahead=1, max_depth=26, node_budget=10000, lookahead_candidate_limit=80),
            "balanced": dict(strategy="bounded-lookahead", lookahead=2, max_depth=30, node_budget=30000, lookahead_candidate_limit=250),
            "deep": dict(strategy="bounded-lookahead", lookahead=3, max_depth=36, node_budget=80000, lookahead_candidate_limit=500),
            "exact": dict(strategy="exact", lookahead=26, max_depth=26, node_budget=200000, lookahead_candidate_limit=1000),
        }[profile]
        if lookahead is not None:
            defaults["lookahead"] = lookahead
        if node_budget is not None:
            defaults["node_budget"] = node_budget
        return cls(length=length, max_lives=max_lives, weighting=weighting, profile=profile, **defaults)


class HangmanTreeBuilder:
    """Construct a multiway Hangman policy tree.

    The produced tree is the player. During inference no frequency, entropy,
    classifier, or external agent is consulted; the stored node supplies the
    next guess and the observed Hangman pattern selects the child branch.
    """

    def __init__(
        self,
        words: list[str],
        weights: dict[str, float] | None = None,
        config: TreeBuildConfig | None = None,
        start_node_id: int = 0,
        reserved_node_ids: set[int] | None = None,
    ) -> None:
        if not words:
            raise ValueError("Cannot build a tree without words.")
        if start_node_id < 0:
            raise ValueError("start_node_id must be non-negative.")
        length = len(words[0])
        if any(len(word) != length for word in words):
            raise ValueError("A word-length-specific tree can only contain one length.")
        self.words = tuple(sorted(set(words)))
        self.word_index = {word: index for index, word in enumerate(self.words)}
        self.config = config or TreeBuildConfig(length=length)
        if self.config.length != length:
            raise ValueError("TreeBuildConfig length does not match vocabulary.")
        self.weights = tuple(self._word_weight(word, weights) for word in self.words)
        self.total_weight = sum(self.weights)
        self.nodes: dict[int, TreeNode] = {}
        self._next_node_id = start_node_id
        self._reserved_node_ids = set(reserved_node_ids or set())
        self._estimate_cache: dict[tuple[tuple[int, ...], str, tuple[str, ...], int, int], TreeStats] = {}
        self._pattern_cache: dict[tuple[int, str], str] = {}
        self.training_notes: list[str] = []

    def build(self, forced_root_letter: str | None = None) -> HangmanDecisionTree:
        started = time.perf_counter()
        root_candidates = tuple(range(len(self.words)))
        root_id = self._build_node(
            candidates=root_candidates,
            pattern=UNKNOWN * self.config.length,
            guessed=frozenset(),
            incorrect=frozenset(),
            remaining_lives=self.config.max_lives,
            depth=0,
            forced_letter=forced_root_letter,
            reserved_slots=0,
        )
        training_seconds = time.perf_counter() - started
        strategy = self.config.strategy
        if strategy == "exact" and len(self.words) > self.config.exact_candidate_limit:
            strategy = "near-exact"
            self.training_notes.append(
                f"Exact optimization was not attempted above {self.config.exact_candidate_limit} candidates."
            )
        tree = HangmanDecisionTree(
            length=self.config.length,
            objective=TreeObjective(
                max_lives=self.config.max_lives,
                weighting=self.config.weighting,
            ),
            strategy=strategy,  # type: ignore[arg-type]
            root_id=root_id,
            nodes=self.nodes,
            metadata={
                "training_seconds": training_seconds,
                "word_count": len(self.words),
                "profile": self.config.profile,
                "requested_strategy": self.config.strategy,
                "lookahead": self.config.lookahead,
                "node_budget": self.config.node_budget,
                "max_depth": self.config.max_depth,
                "lookahead_candidate_limit": self.config.lookahead_candidate_limit,
                "exact_candidate_limit": self.config.exact_candidate_limit,
                "pruning_min_gain": self.config.pruning_min_gain,
                "weighting": self.config.weighting,
                "tree_format_version": TREE_FORMAT_VERSION,
                "training_notes": self.training_notes,
            },
        )
        return tree

    def build_from_state(
        self,
        *,
        candidates: tuple[str, ...] | list[str],
        pattern: str,
        guessed_letters: frozenset[str] | set[str] | tuple[str, ...] | list[str],
        incorrect_letters: frozenset[str] | set[str] | tuple[str, ...] | list[str],
        remaining_lives: int,
        depth: int,
        root_node_id: int | None = None,
    ) -> HangmanDecisionTree:
        if len(pattern) != self.config.length:
            raise ValueError("Subtree pattern length does not match TreeBuildConfig length.")
        if root_node_id is not None:
            if root_node_id < 0:
                raise ValueError("root_node_id must be non-negative.")
            self._next_node_id = root_node_id
            self._reserved_node_ids.discard(root_node_id)

        started = time.perf_counter()
        unknown_words = sorted(set(candidates) - set(self.word_index))
        if unknown_words:
            raise ValueError(f"Subtree candidates are not present in the builder vocabulary: {unknown_words[:3]}")
        candidate_ids = tuple(self.word_index[word] for word in sorted(set(candidates)))
        guessed = frozenset(guessed_letters)
        incorrect = frozenset(incorrect_letters)
        if not incorrect.issubset(guessed):
            raise ValueError("Incorrect letters must also be present in guessed_letters.")

        root_id = self._build_node(
            candidates=candidate_ids,
            pattern=pattern,
            guessed=guessed,
            incorrect=incorrect,
            remaining_lives=remaining_lives,
            depth=depth,
            reserved_slots=0,
        )
        training_seconds = time.perf_counter() - started
        strategy = self.config.strategy
        if strategy == "exact" and len(candidate_ids) > self.config.exact_candidate_limit:
            strategy = "near-exact"
            self._add_training_note(
                f"Exact optimization was not attempted above {self.config.exact_candidate_limit} candidates."
            )
        return HangmanDecisionTree(
            length=self.config.length,
            objective=TreeObjective(
                max_lives=self.config.max_lives,
                weighting=self.config.weighting,
            ),
            strategy=strategy,  # type: ignore[arg-type]
            root_id=root_id,
            nodes=self.nodes,
            metadata={
                "training_seconds": training_seconds,
                "word_count": len(candidate_ids),
                "profile": self.config.profile,
                "requested_strategy": self.config.strategy,
                "lookahead": self.config.lookahead,
                "node_budget": self.config.node_budget,
                "max_depth": self.config.max_depth,
                "lookahead_candidate_limit": self.config.lookahead_candidate_limit,
                "exact_candidate_limit": self.config.exact_candidate_limit,
                "pruning_min_gain": self.config.pruning_min_gain,
                "weighting": self.config.weighting,
                "tree_format_version": TREE_FORMAT_VERSION,
                "training_notes": self.training_notes,
                "root_pattern": pattern,
                "root_guessed_letters": sorted(guessed),
                "root_incorrect_letters": sorted(incorrect),
                "root_remaining_lives": remaining_lives,
                "root_depth": depth,
            },
        )

    def _word_weight(self, word: str, weights: dict[str, float] | None) -> float:
        if self.config.weighting == "uniform" or not weights:
            return 1.0
        return max(float(weights.get(word, 1.0)), 0.0)

    def _new_node_id(self) -> int:
        while self._next_node_id in self._reserved_node_ids:
            self._next_node_id += 1
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def _can_allocate(self, node_count: int, reserved_slots: int) -> bool:
        return len(self.nodes) + node_count <= self.config.node_budget - reserved_slots

    def _add_training_note(self, note: str) -> None:
        if note not in self.training_notes:
            self.training_notes.append(note)

    def _state_hash(self, candidates: tuple[int, ...]) -> str:
        digest = hashlib.blake2b(digest_size=12)
        for candidate in candidates:
            digest.update(candidate.to_bytes(4, "little", signed=False))
        return digest.hexdigest()

    def _pattern(self, word_index: int, letter: str) -> str:
        key = (word_index, letter)
        if key not in self._pattern_cache:
            self._pattern_cache[key] = outcome_for_word(self.words[word_index], letter)
        return self._pattern_cache[key]

    def _build_node(
        self,
        candidates: tuple[int, ...],
        pattern: str,
        guessed: frozenset[str],
        incorrect: frozenset[str],
        remaining_lives: int,
        depth: int,
        forced_letter: str | None = None,
        reserved_slots: int = 0,
    ) -> int:
        if not self._can_allocate(1, reserved_slots):
            raise RuntimeError("Node budget exhausted before reserving required tree branches.")
        node_id = self._new_node_id()

        terminal_leaf = self._terminal_leaf_type(pattern, candidates, remaining_lives, guessed)
        if terminal_leaf:
            node = self._make_leaf(
                node_id,
                depth,
                pattern,
                guessed,
                incorrect,
                remaining_lives,
                candidates,
                terminal_leaf,
            )
            self.nodes[node_id] = node
            return node_id

        if depth >= self.config.max_depth:
            node = self._make_leaf(
                node_id,
                depth,
                pattern,
                guessed,
                incorrect,
                remaining_lives,
                candidates,
                "depth_limited",
            )
            self.nodes[node_id] = node
            return node_id

        letter = forced_letter or self._choose_letter(candidates, pattern, guessed, remaining_lives)
        if letter is None:
            node = self._make_leaf(
                node_id,
                depth,
                pattern,
                guessed,
                incorrect,
                remaining_lives,
                candidates,
                "exhausted",
            )
            self.nodes[node_id] = node
            return node_id

        partitions = self._partition(candidates, letter)
        if not self._can_allocate(1 + len(partitions), reserved_slots):
            self._add_training_note(
                "Node budget reached; expandable budget leaves preserve the exact Hangman state."
            )
            node = self._make_leaf(
                node_id,
                depth,
                pattern,
                guessed,
                incorrect,
                remaining_lives,
                candidates,
                "budget",
            )
            self.nodes[node_id] = node
            return node_id

        if self.config.pruning_min_gain > 0 and forced_letter is None:
            fallback_stats = self._leaf_stats(
                pattern,
                candidates,
                remaining_lives,
                guessed,
                "depth_limited",
            )
            split_stats = self._estimate_after_letter(
                candidates,
                pattern,
                guessed,
                remaining_lives,
                letter,
                1,
            )
            if split_stats.win_probability - fallback_stats.win_probability < self.config.pruning_min_gain:
                note = (
                    f"Pruned nodes whose immediate win-rate gain was below "
                    f"{self.config.pruning_min_gain}."
                )
                self._add_training_note(note)
                node = self._make_leaf(
                    node_id,
                    depth,
                    pattern,
                    guessed,
                    incorrect,
                    remaining_lives,
                    candidates,
                    "depth_limited",
                )
                self.nodes[node_id] = node
                return node_id

        node = TreeNode(
            node_id=node_id,
            depth=depth,
            pattern=pattern,
            guessed_letters=tuple(sorted(guessed)),
            incorrect_letters=tuple(sorted(incorrect)),
            remaining_lives=remaining_lives,
            candidate_count=len(candidates),
            guess=letter,
            candidate_sample=tuple(self.words[index] for index in candidates[: self.config.sample_size]),
            fallback_letters=self._fallback_letters(candidates, guessed),
        )
        self.nodes[node_id] = node

        partition_items = sorted(partitions.items())
        branch_stats: list[tuple[float, TreeStats]] = []
        for index, (outcome_pattern, child_candidates) in enumerate(partition_items):
            is_miss = "1" not in outcome_pattern
            child_lives = remaining_lives - 1 if is_miss else remaining_lives
            child_pattern = apply_outcome(pattern, letter, outcome_pattern)
            child_incorrect = incorrect | frozenset({letter}) if is_miss else incorrect
            sibling_slots = len(partition_items) - index - 1
            child_id = self._build_node(
                candidates=child_candidates,
                pattern=child_pattern,
                guessed=guessed | frozenset({letter}),
                incorrect=child_incorrect,
                remaining_lives=child_lives,
                depth=depth + 1,
                reserved_slots=reserved_slots + sibling_slots,
            )
            probability = self._candidate_weight(child_candidates) / max(self._candidate_weight(candidates), 1e-12)
            branch = BranchInfo(
                outcome_pattern=outcome_pattern,
                node_id=child_id,
                word_count=len(child_candidates),
                probability=probability,
                is_miss=is_miss,
                resulting_pattern=child_pattern,
                remaining_lives=child_lives,
            )
            node.children[outcome_pattern] = child_id
            node.branches[outcome_pattern] = branch
            branch_stats.append((probability, self._stats_for_node(self.nodes[child_id])))

        aggregate = self._aggregate_child_stats(branch_stats)
        node.win_probability = aggregate.win_probability
        node.average_mistakes = aggregate.average_mistakes + self._miss_probability(partitions, candidates, letter)
        node.average_total_guesses = aggregate.average_total_guesses + 1.0
        node.average_remaining_lives = aggregate.average_remaining_lives
        node.expected_depth = aggregate.expected_depth + 1.0
        return node_id

    def _terminal_leaf_type(
        self,
        pattern: str,
        candidates: tuple[int, ...],
        remaining_lives: int,
        guessed: frozenset[str],
    ) -> str | None:
        if is_solved_pattern(pattern):
            return "solved"
        if remaining_lives <= 0 or not candidates:
            return "loss"
        if not any(letter not in guessed for letter in ALPHABET):
            return "exhausted"
        return None

    def _make_leaf(
        self,
        node_id: int,
        depth: int,
        pattern: str,
        guessed: frozenset[str],
        incorrect: frozenset[str],
        remaining_lives: int,
        candidates: tuple[int, ...],
        leaf_type: str,
    ) -> TreeNode:
        stats = self._leaf_stats(pattern, candidates, remaining_lives, guessed, leaf_type)
        return TreeNode(
            node_id=node_id,
            depth=depth,
            pattern=pattern,
            guessed_letters=tuple(sorted(guessed)),
            incorrect_letters=tuple(sorted(incorrect)),
            remaining_lives=remaining_lives,
            candidate_count=len(candidates),
            leaf_type=leaf_type,  # type: ignore[arg-type]
            win_probability=stats.win_probability,
            average_mistakes=stats.average_mistakes,
            average_total_guesses=stats.average_total_guesses,
            average_remaining_lives=stats.average_remaining_lives,
            expected_depth=stats.expected_depth,
            candidate_sample=tuple(self.words[index] for index in candidates[: self.config.sample_size]),
            fallback_letters=self._fallback_letters(candidates, guessed),
        )

    def _leaf_stats(
        self,
        pattern: str,
        candidates: tuple[int, ...],
        remaining_lives: int,
        guessed: frozenset[str],
        leaf_type: str,
    ) -> TreeStats:
        if leaf_type == "solved":
            return TreeStats(1.0, 0.0, 0.0, remaining_lives, 0.0, 1, 1, 0)
        if leaf_type == "loss" or remaining_lives <= 0 or not candidates:
            return TreeStats(0.0, 0.0, 0.0, remaining_lives, 0.0, 1, 1, 0)

        # Budget/depth leaves contain a fixed stored guess order. Estimate the
        # quality of that stored policy without consulting any external solver.
        wins = 0.0
        total_weight = self._candidate_weight(candidates)
        mistakes = 0.0
        total_guesses = 0.0
        remaining_total = 0.0
        fallback = self._fallback_letters(candidates, guessed)
        for word_index in candidates:
            word = self.words[word_index]
            lives = remaining_lives
            local_guessed = set(guessed)
            local_pattern = pattern
            turns = 0
            wrong = 0
            for letter in fallback:
                if UNKNOWN not in local_pattern or lives <= 0:
                    break
                if letter in local_guessed:
                    continue
                local_guessed.add(letter)
                turns += 1
                if letter in word:
                    local_pattern = apply_outcome(local_pattern, letter, self._pattern(word_index, letter))
                else:
                    lives -= 1
                    wrong += 1
            weight = self.weights[word_index] / max(total_weight, 1e-12)
            wins += weight * float(UNKNOWN not in local_pattern and lives > 0)
            mistakes += weight * wrong
            total_guesses += weight * turns
            remaining_total += weight * lives
        return TreeStats(wins, mistakes, total_guesses, remaining_total, 0.0, 1, 1, 0)

    def _stats_for_node(self, node: TreeNode) -> TreeStats:
        return TreeStats(
            win_probability=node.win_probability,
            average_mistakes=node.average_mistakes,
            average_total_guesses=node.average_total_guesses,
            average_remaining_lives=node.average_remaining_lives,
            expected_depth=node.expected_depth,
            node_count=1,
            leaf_count=1 if node.is_leaf else 0,
            max_depth=node.depth,
        )

    def _aggregate_child_stats(self, branch_stats: list[tuple[float, TreeStats]]) -> TreeStats:
        if not branch_stats:
            return TreeStats(0.0, 0.0, 0.0, 0.0, 1.0, 1, 1, 0)
        return TreeStats(
            win_probability=sum(prob * stats.win_probability for prob, stats in branch_stats),
            average_mistakes=sum(prob * stats.average_mistakes for prob, stats in branch_stats),
            average_total_guesses=sum(prob * stats.average_total_guesses for prob, stats in branch_stats),
            average_remaining_lives=sum(prob * stats.average_remaining_lives for prob, stats in branch_stats),
            expected_depth=sum(prob * stats.expected_depth for prob, stats in branch_stats),
            node_count=1 + sum(stats.node_count for _, stats in branch_stats),
            leaf_count=sum(stats.leaf_count for _, stats in branch_stats),
            max_depth=1 + max(stats.max_depth for _, stats in branch_stats),
        )

    def _candidate_weight(self, candidates: tuple[int, ...]) -> float:
        return sum(self.weights[index] for index in candidates)

    def _partition(self, candidates: tuple[int, ...], letter: str) -> dict[str, tuple[int, ...]]:
        groups: dict[str, list[int]] = defaultdict(list)
        for word_index in candidates:
            groups[self._pattern(word_index, letter)].append(word_index)
        return {pattern: tuple(indices) for pattern, indices in groups.items()}

    def _choose_letter(
        self,
        candidates: tuple[int, ...],
        pattern: str,
        guessed: frozenset[str],
        remaining_lives: int,
    ) -> str | None:
        available = [letter for letter in ALPHABET if letter not in guessed]
        if not available:
            return None
        if len(candidates) == 1:
            word = self.words[candidates[0]]
            return next((letter for letter in word if letter not in guessed), available[0])

        scored = []
        for letter in available:
            if (
                self.config.strategy in {"bounded-lookahead", "exact"}
                and self.config.lookahead > 1
                and len(candidates) <= self.config.lookahead_candidate_limit
            ):
                stats = self._estimate_after_letter(
                    candidates,
                    pattern,
                    guessed,
                    remaining_lives,
                    letter,
                    self.config.lookahead,
                )
                scored.append((stats.score_tuple(), letter))
            else:
                scored.append((self._greedy_score(candidates, remaining_lives, letter), letter))
        scored.sort(reverse=True)
        return scored[0][1]

    def _greedy_score(
        self,
        candidates: tuple[int, ...],
        remaining_lives: int,
        letter: str,
    ) -> tuple[float, float, float, float, str]:
        partitions = self._partition(candidates, letter)
        total_weight = self._candidate_weight(candidates)
        miss_pattern = "0" * self.config.length
        miss_weight = self._candidate_weight(partitions.get(miss_pattern, tuple()))
        hit_probability = 1.0 - miss_weight / max(total_weight, 1e-12)
        expected_child_size = sum(
            (self._candidate_weight(child) / max(total_weight, 1e-12)) * len(child)
            for child in partitions.values()
        )
        branch_count = len(partitions)
        life_pressure = (self.config.max_lives + 1) / max(remaining_lives + 1, 1)
        score = (
            hit_probability,
            -miss_weight / max(total_weight, 1e-12) * life_pressure,
            -expected_child_size,
            branch_count,
            letter,
        )
        return score

    def _estimate_after_letter(
        self,
        candidates: tuple[int, ...],
        pattern: str,
        guessed: frozenset[str],
        remaining_lives: int,
        letter: str,
        lookahead: int,
    ) -> TreeStats:
        partitions = self._partition(candidates, letter)
        total = self._candidate_weight(candidates)
        branch_stats = []
        for outcome_pattern, child_candidates in partitions.items():
            is_miss = "1" not in outcome_pattern
            child_lives = remaining_lives - 1 if is_miss else remaining_lives
            child_pattern = apply_outcome(pattern, letter, outcome_pattern)
            child_stats = self._estimate_state(
                child_candidates,
                child_pattern,
                guessed | frozenset({letter}),
                child_lives,
                lookahead - 1,
            )
            probability = self._candidate_weight(child_candidates) / max(total, 1e-12)
            branch_stats.append((probability, child_stats))
        aggregate = self._aggregate_child_stats(branch_stats)
        return TreeStats(
            win_probability=aggregate.win_probability,
            average_mistakes=aggregate.average_mistakes + self._miss_probability(partitions, candidates, letter),
            average_total_guesses=aggregate.average_total_guesses + 1.0,
            average_remaining_lives=aggregate.average_remaining_lives,
            expected_depth=aggregate.expected_depth + 1.0,
            node_count=aggregate.node_count + 1,
            leaf_count=aggregate.leaf_count,
            max_depth=aggregate.max_depth + 1,
        )

    def _estimate_state(
        self,
        candidates: tuple[int, ...],
        pattern: str,
        guessed: frozenset[str],
        remaining_lives: int,
        lookahead: int,
    ) -> TreeStats:
        leaf_type = self._terminal_leaf_type(pattern, candidates, remaining_lives, guessed)
        if leaf_type or lookahead <= 0:
            return self._leaf_stats(
                pattern,
                candidates,
                remaining_lives,
                guessed,
                leaf_type or "depth_limited",
            )
        key = (
            candidates,
            pattern,
            tuple(sorted(guessed)),
            remaining_lives,
            lookahead,
        )
        if key in self._estimate_cache:
            return self._estimate_cache[key]
        available = [letter for letter in ALPHABET if letter not in guessed]
        best = max(
            (
                self._estimate_after_letter(
                    candidates,
                    pattern,
                    guessed,
                    remaining_lives,
                    letter,
                    lookahead,
                )
                for letter in available
            ),
            key=lambda stats: stats.score_tuple(),
        )
        self._estimate_cache[key] = best
        return best

    def _miss_probability(
        self,
        partitions: dict[str, tuple[int, ...]],
        candidates: tuple[int, ...],
        letter: str,
    ) -> float:
        miss_pattern = "0" * self.config.length
        return self._candidate_weight(partitions.get(miss_pattern, tuple())) / max(
            self._candidate_weight(candidates),
            1e-12,
        )

    def _fallback_letters(self, candidates: tuple[int, ...], guessed: frozenset[str]) -> tuple[str, ...]:
        counts: Counter[str] = Counter()
        for word_index in candidates:
            counts.update(set(self.words[word_index]) - set(guessed))
        if len(candidates) == 1:
            word = self.words[candidates[0]]
            word_letters = tuple(dict.fromkeys(letter for letter in word if letter not in guessed))
            remaining = tuple(
                letter
                for letter in ALPHABET
                if letter not in guessed and letter not in word_letters
            )
            return word_letters + remaining
        ranked = sorted(
            (letter for letter in ALPHABET if letter not in guessed),
            key=lambda letter: (-counts[letter], letter),
        )
        return tuple(ranked)
