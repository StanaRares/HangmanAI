from __future__ import annotations

import pytest

from app.core.environment import HangmanEnvironment


def test_repeated_letters_are_revealed_together() -> None:
    environment = HangmanEnvironment(["apple"], max_lives=6)
    state = environment.reset("apple")
    assert state.pattern == "_____"

    state = environment.step("p")
    assert state.pattern == "_pp__"
    assert state.remaining_lives == 6
    assert state.history[-1].positions == (1, 2)


def test_wrong_guess_reduces_life_and_records_history() -> None:
    environment = HangmanEnvironment(["apple"], max_lives=6)
    environment.reset("apple")
    state = environment.step("z")
    assert state.pattern == "_____"
    assert state.remaining_lives == 5
    assert state.incorrect_letters == frozenset({"z"})
    assert state.history[-1].correct is False


def test_repeated_guess_is_invalid() -> None:
    environment = HangmanEnvironment(["apple"])
    environment.reset("apple")
    environment.step("a")
    with pytest.raises(ValueError):
        environment.step("a")


def test_win_and_loss_status() -> None:
    win_env = HangmanEnvironment(["ab"], max_lives=2)
    win_env.reset("ab")
    win_env.step("a")
    state = win_env.step("b")
    assert state.status == "won"

    loss_env = HangmanEnvironment(["ab"], max_lives=1)
    loss_env.reset("ab")
    state = loss_env.step("z")
    assert state.status == "lost"
