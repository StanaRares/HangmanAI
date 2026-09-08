import type { GameState } from "../types";

const LETTERS = "abcdefghijklmnopqrstuvwxyz".split("");

type Props = {
  state?: GameState;
  onGuess: (letter: string) => void;
  disabled?: boolean;
  highlightedLetter?: string;
};

export function LetterKeyboard({ state, onGuess, disabled = false, highlightedLetter }: Props) {
  const guessed = new Set(state?.guessed_letters ?? []);
  const wrong = new Set(state?.incorrect_letters ?? []);
  const correct = new Set(state?.correct_letters ?? []);
  const highlighted = highlightedLetter?.toLowerCase();

  return (
    <div className="keyboard" aria-label="Alphabet keyboard">
      {LETTERS.map((letter) => {
        const used = guessed.has(letter);
        return (
          <button
            key={letter}
            type="button"
            className={`key ${used ? "used" : ""} ${correct.has(letter) ? "correct" : ""} ${wrong.has(letter) ? "wrong" : ""} ${highlighted === letter ? "selected" : ""}`}
            onClick={() => onGuess(letter)}
            disabled={disabled || used || state?.status !== "playing"}
            title={`Guess ${letter.toUpperCase()}`}
          >
            {letter.toUpperCase()}
          </button>
        );
      })}
    </div>
  );
}
