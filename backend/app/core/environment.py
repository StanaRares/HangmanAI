from __future__ import annotations

import random
from collections.abc import Sequence

from app.core.dictionary import clean_word
from app.core.state import GuessRecord, GameState, UNKNOWN, normalize_letter


class HangmanEnvironment:
    """Reusable Hangman environment with a hidden word and public state."""

    def __init__(
        self,
        words: Sequence[str],
        max_lives: int = 6,
        seed: int | None = None,
    ) -> None:
        if max_lives < 1:
            raise ValueError("max_lives must be at least 1.")
        self.words = tuple(clean_word(word) for word in words if clean_word(word))
        if not self.words:
            raise ValueError("At least one valid word is required.")
        self.max_lives = max_lives
        self._rng = random.Random(seed)
        self._word: str | None = None
        self._state: GameState | None = None

    @property
    def solution(self) -> str:
        if self._word is None:
            raise RuntimeError("Environment has not been reset.")
        return self._word

    def reset(self, word: str | None = None) -> GameState:
        if word is None:
            selected = self._rng.choice(self.words)
        else:
            selected = clean_word(word)
            if not selected:
                raise ValueError("Provided word must contain only letters.")

        self._word = selected
        self._state = GameState(
            word_length=len(selected),
            pattern=UNKNOWN * len(selected),
            guessed_letters=frozenset(),
            incorrect_letters=frozenset(),
            remaining_lives=self.max_lives,
            max_lives=self.max_lives,
        )
        return self._state

    def get_state(self) -> GameState:
        if self._state is None:
            return self.reset()
        return self._state

    def is_finished(self) -> bool:
        return self.get_state().is_finished

    def step(self, letter: str) -> GameState:
        state = self.get_state()
        if state.is_finished:
            raise ValueError("Cannot guess after the game has finished.")

        guess = normalize_letter(letter)
        if guess in state.guessed_letters:
            raise ValueError(f"Letter '{guess}' has already been guessed.")

        word = self.solution
        positions = tuple(index for index, char in enumerate(word) if char == guess)
        pattern_chars = list(state.pattern)
        for index in positions:
            pattern_chars[index] = guess

        correct = bool(positions)
        remaining_lives = state.remaining_lives if correct else state.remaining_lives - 1
        pattern = "".join(pattern_chars)
        status = "playing"
        if UNKNOWN not in pattern:
            status = "won"
        elif remaining_lives <= 0:
            status = "lost"

        record = GuessRecord(
            letter=guess,
            correct=correct,
            positions=positions,
            pattern=pattern,
            remaining_lives=remaining_lives,
        )
        self._state = GameState(
            word_length=state.word_length,
            pattern=pattern,
            guessed_letters=state.guessed_letters | frozenset({guess}),
            incorrect_letters=(
                state.incorrect_letters if correct else state.incorrect_letters | frozenset({guess})
            ),
            remaining_lives=remaining_lives,
            max_lives=state.max_lives,
            status=status,
            turn=state.turn + 1,
            history=state.history + (record,),
        )
        return self._state
