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
      <svg viewBox="0 0 220 220" role="img">
        <path className="gallows" d="M42 198h126M70 198V32h92v34" />
        <path className="rope" d="M162 66v22" />
        {parts.map((part, index) => (
          <path
            key={part}
            className={`body-part ${misses > index ? "visible" : ""}`}
            d={
              part === "head"
                ? "M144 108a18 18 0 1 0 36 0a18 18 0 1 0 -36 0"
                : part === "body"
                  ? "M162 126v44"
                  : part === "left-arm"
                    ? "M162 138l-28 18"
                    : part === "right-arm"
                      ? "M162 138l28 18"
                      : part === "left-leg"
                        ? "M162 170l-26 28"
                        : "M162 170l26 28"
            }
          />
        ))}
      </svg>
    </div>
  );
}
