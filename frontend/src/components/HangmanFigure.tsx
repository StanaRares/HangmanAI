import type { GameState } from "../types";

type Props = {
  state?: GameState;
};

export function HangmanFigure({ state }: Props) {
  const misses = state ? state.max_lives - state.remaining_lives : 0;
  const parts = [
    "head",
    "body",
    "left-arm",
    "right-arm",
    "left-leg",
    "right-leg"
  ];

  return (
    <div className="hangman-figure" aria-label="Hangman drawing">
      <svg viewBox="20 40 224 220" preserveAspectRatio="xMidYMin meet" role="img">
        <path className="gallows gallows-base" d="M34 220c38 -3 83 -3 145 0" />
        <path className="gallows gallows-post" d="M76 219c-2 -54 -1 -107 3 -161" />
        <path className="gallows gallows-beam" d="M76 60c41 -7 79 -7 116 1" />
        <path className="rope" d="M190 60c-1 15 -2 29 -2 42" />
        {parts.map((part, index) => (
          <path
            key={part}
            className={`body-part ${misses > index ? "visible" : ""}`}
            d={
              part === "head"
                ? "M168 124c3 -19 24 -25 36 -10c11 16 0 36 -18 35c-15 -1 -22 -13 -18 -25"
                : part === "body"
                  ? "M186 150c-3 23 -3 39 1 58"
                  : part === "left-arm"
                    ? "M184 166c-18 8 -31 18 -44 31"
                    : part === "right-arm"
                      ? "M188 166c17 9 31 20 43 34"
                      : part === "left-leg"
                        ? "M187 207c-17 14 -29 28 -40 43"
                        : "M188 207c15 14 29 28 42 43"
            }
          />
        ))}
      </svg>
    </div>
  );
}
