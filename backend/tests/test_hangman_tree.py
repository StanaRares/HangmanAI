from __future__ import annotations

import pytest

from app.core.state import GameState
from app.tree.builder import HangmanTreeBuilder, TreeBuildConfig
from app.tree.evaluation import evaluate_tree, evaluate_tree_on_word
from app.tree.model import canonical_state_key
from app.tree.serialization import load_tree, save_tree


def config(length: int, **overrides) -> TreeBuildConfig:
    return TreeBuildConfig.from_profile(
        length=length,
        profile="fast",
        node_budget=5000,
        **overrides,
    )


def test_multiway_branching_for_letter_positions() -> None:
    tree = HangmanTreeBuilder(["cat", "car", "can", "dog"], config(3)).build(
        forced_root_letter="a"
    )

    root = tree.root
    assert root.guess == "a"
    assert set(root.branches) == {"010", "000"}
    assert root.branches["010"].word_count == 3
    assert root.branches["000"].word_count == 1
    assert root.branches["010"].resulting_pattern == "_a_"


def test_repeated_letters_create_distinct_position_branches() -> None:
    tree = HangmanTreeBuilder(["apple", "ample"], config(5)).build(
        forced_root_letter="p"
    )

    root = tree.root
    assert root.branches["01100"].word_count == 1
    assert root.branches["00100"].word_count == 1
    assert root.branches["01100"].resulting_pattern == "_pp__"
    assert root.branches["00100"].resulting_pattern == "__p__"


def test_equivalent_states_hash_identically() -> None:
    left = canonical_state_key(
        length=5,
        pattern="a___e",
        guessed_letters={"e", "a"},
        incorrect_letters={"z", "q"},
        remaining_lives=4,
        candidates=["apple", "angle"],
    )
    right = canonical_state_key(
        length=5,
        pattern="a___e",
        guessed_letters={"a", "e"},
        incorrect_letters={"q", "z"},
        remaining_lives=4,
        candidates=["angle", "apple"],
    )

    assert left == right


def test_tree_traversal_follows_observed_child_branches() -> None:
    tree = HangmanTreeBuilder(["cat", "car"], config(3)).build(forced_root_letter="a")
    root = tree.root
    next_node = tree.next_node_id(root.node_id, "010")

    result = evaluate_tree_on_word(tree, "cat")

    assert next_node is not None
    assert result.node_sequence[:2] == (root.node_id, next_node)
    assert result.won


def test_evaluation_covers_every_word_in_tiny_vocabulary() -> None:
    words = ["cat", "car", "can", "dog"]
    tree = HangmanTreeBuilder(words, config(3)).build()

    summary, games = evaluate_tree(tree, words)

    assert summary["total_words"] == len(words)
    assert len(games) == len(words)
    assert all(row["status"] in {"won", "lost"} for row in games)


def test_tree_serialization_round_trips_decisions(tmp_path) -> None:
    tree = HangmanTreeBuilder(["cat", "car", "can"], config(3)).build()
    path = tmp_path / "tree.json.gz"
    save_tree(tree, path)
    loaded = load_tree(path)
    state = GameState(
        word_length=3,
        pattern="___",
        guessed_letters=frozenset(),
        incorrect_letters=frozenset(),
        remaining_lives=6,
        max_lives=6,
    )

    assert loaded.best_first_guess == tree.best_first_guess
    assert loaded.next_guess(loaded.root_id, state) == tree.next_guess(tree.root_id, state)
    assert loaded.node_count == tree.node_count


def test_word_length_specific_trees_reject_mixed_lengths() -> None:
    with pytest.raises(ValueError):
        HangmanTreeBuilder(["apple", "planet"], config(5)).build()
