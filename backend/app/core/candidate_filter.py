from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping

from app.core.state import ALPHABET, GameState, UNKNOWN


def letter_pattern(word: str, letter: str) -> str:
    """Return a binary occurrence pattern for a guessed letter in a word."""
    return "".join("1" if char == letter else "0" for char in word)


class CandidateIndex:
    """Length-indexed dictionary with Hangman-compatible filtering."""

    def __init__(
        self,
        words: Iterable[str],
        weights: Mapping[str, float] | None = None,
    ) -> None:
        buckets: dict[int, list[str]] = defaultdict(list)
        unique_words = sorted(set(words))
        for word in unique_words:
            buckets[len(word)].append(word)

        self.words_by_length: dict[int, tuple[str, ...]] = {
            length: tuple(sorted(bucket)) for length, bucket in buckets.items()
        }
        self.words: tuple[str, ...] = tuple(
            word for length in sorted(self.words_by_length) for word in self.words_by_length[length]
        )
        self.letter_masks: dict[str, int] = {
            word: self._letter_mask(word) for word in self.words
        }
        self.weights = {word: float(weights[word]) for word in self.words if weights and word in weights}
        self._candidate_cache: dict[tuple[int, str, tuple[str, ...]], tuple[str, ...]] = {}

    def words_of_length(self, length: int) -> tuple[str, ...]:
        return self.words_by_length.get(length, tuple())

    def weight(self, word: str) -> float:
        return max(self.weights.get(word, 1.0), 0.0)

    @staticmethod
    def _letter_mask(word: str) -> int:
        mask = 0
        for char in set(word):
            if char in ALPHABET:
                mask |= 1 << ALPHABET.index(char)
        return mask

    def candidates(self, state: GameState) -> tuple[str, ...]:
        key = (state.word_length, state.pattern, tuple(sorted(state.guessed_letters)))
        if key not in self._candidate_cache:
            self._candidate_cache[key] = tuple(
                word
                for word in self.words_of_length(state.word_length)
                if self.is_compatible(
                    word=word,
                    pattern=state.pattern,
                    guessed_letters=state.guessed_letters,
                )
            )
        return self._candidate_cache[key]

    @staticmethod
    def is_compatible(
        word: str,
        pattern: str,
        guessed_letters: frozenset[str] | set[str],
    ) -> bool:
        if len(word) != len(pattern):
            return False

        for index, pattern_char in enumerate(pattern):
            if pattern_char != UNKNOWN and word[index] != pattern_char:
                return False

        # Every occurrence of a guessed letter is revealed by Hangman. A candidate
        # with an extra guessed letter in an unknown slot is therefore impossible.
        for letter in guessed_letters:
            word_positions = {index for index, char in enumerate(word) if char == letter}
            revealed_positions = {
                index for index, char in enumerate(pattern) if char == letter
            }
            if word_positions != revealed_positions:
                return False

        return True

    def letter_document_counts(self, candidates: Iterable[str]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for word in candidates:
            mask = self.letter_masks.get(word, self._letter_mask(word))
            for index, letter in enumerate(ALPHABET):
                if mask & (1 << index):
                    counts[letter] += 1
        return counts

    def letter_probabilities(self, candidates: tuple[str, ...]) -> dict[str, float]:
        if not candidates:
            return {letter: 0.0 for letter in ALPHABET}
        counts = self.letter_document_counts(candidates)
        total = len(candidates)
        return {letter: counts[letter] / total for letter in ALPHABET}
