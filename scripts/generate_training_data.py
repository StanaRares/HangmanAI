from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.agents.entropy_agent import EntropyAgent  # noqa: E402
from app.core.candidate_filter import CandidateIndex  # noqa: E402
from app.core.dictionary import PROJECT_ROOT, read_words  # noqa: E402
from app.core.environment import HangmanEnvironment  # noqa: E402
from app.ml.features import extract_feature_dict  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate supervised Hangman states using an entropy teacher."
    )
    parser.add_argument(
        "--dictionary",
        default=str(PROJECT_ROOT / "data" / "processed" / "train.txt"),
        help="Solver dictionary used for candidate filtering.",
    )
    parser.add_argument(
        "--words",
        default=None,
        help="Optional hidden-word list. Defaults to --dictionary.",
    )
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-lives", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument(
        "--teacher-scoring",
        choices=["entropy_only", "risk_adjusted_entropy"],
        default="risk_adjusted_entropy",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "processed" / "training_states.parquet"),
    )
    parser.add_argument(
        "--partitioned",
        action="store_true",
        help=(
            "Write part-*.parquet files into --output as a directory. "
            "This is the recommended large-experiment mode and supports resume."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partitioned dataset using generation_state.json.",
    )
    return parser.parse_args()


class DatasetWriter:
    def __init__(self, output: Path, partitioned: bool, resume: bool) -> None:
        self.output = output
        self.partitioned = partitioned or output.suffix == ""
        self.writer: pq.ParquetWriter | None = None
        self.part_index = 0
        if self.partitioned:
            output.mkdir(parents=True, exist_ok=True)
            if resume:
                self.part_index = len(sorted(output.glob("part-*.parquet")))
            elif any(output.glob("part-*.parquet")):
                raise SystemExit(f"{output} already contains parts. Use --resume or remove it.")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            if resume and output.exists():
                raise SystemExit("--resume requires --partitioned for scalable appends.")

    def write(self, rows: list[dict[str, object]]) -> int:
        if not rows:
            return 0
        frame = pd.DataFrame(rows)
        if self.output.suffix == ".csv" and not self.partitioned:
            frame.to_csv(
                self.output,
                index=False,
                mode="a" if self.output.exists() else "w",
                header=not self.output.exists(),
            )
            return len(frame)

        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.partitioned:
            part_path = self.output / f"part-{self.part_index:05d}.parquet"
            pq.write_table(table, part_path, compression="zstd")
            self.part_index += 1
        else:
            if self.writer is None:
                self.writer = pq.ParquetWriter(self.output, table.schema, compression="zstd")
            self.writer.write_table(table)
        return len(frame)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def _checkpoint_path(output: Path) -> Path:
    return output / "generation_state.json"


def _load_completed_games(output: Path, resume: bool) -> int:
    if not resume:
        return 0
    checkpoint = _checkpoint_path(output)
    if not checkpoint.exists():
        return 0
    import json

    return int(json.loads(checkpoint.read_text(encoding="utf-8")).get("completed_games", 0))


def _write_checkpoint(output: Path, completed_games: int, rows_written: int) -> None:
    import json

    _checkpoint_path(output).write_text(
        json.dumps(
            {
                "completed_games": completed_games,
                "rows_written": rows_written,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _hidden_word_hash(word: str) -> str:
    return hashlib.sha256(word.encode("utf-8")).hexdigest()


def _write_frame(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".csv":
        frame.to_csv(output, index=False)
    else:
        frame.to_parquet(output, index=False)


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main() -> None:
    args = parse_args()
    if args.games < 1:
        raise SystemExit("--games must be positive.")

    dictionary = read_words(args.dictionary)
    hidden_words = read_words(args.words) if args.words else dictionary
    index = CandidateIndex(dictionary)
    teacher = EntropyAgent(scoring=args.teacher_scoring)
    environment = HangmanEnvironment(hidden_words, max_lives=args.max_lives, seed=args.seed)
    output = Path(args.output)
    writer = DatasetWriter(output, args.partitioned, args.resume)
    completed_games = _load_completed_games(output, args.resume) if writer.partitioned else 0
    rows: list[dict[str, object]] = []
    rows_written = 0

    for game_number in range(completed_games, args.games):
        import random

        word = random.Random(args.seed + game_number).choice(hidden_words)
        state = environment.reset(word)
        while not state.is_finished:
            candidates = index.candidates(state)
            decision = teacher.guess(state, index)
            row = dict(extract_feature_dict(state, candidates, index))
            row.update(
                {
                    "label": decision.letter,
                    "game": game_number,
                    "turn_in_game": state.turn,
                    "hidden_word": word,
                    "hidden_word_hash": _hidden_word_hash(word),
                    "hidden_word_length": len(word),
                    "candidate_count_before": len(candidates),
                    "teacher": teacher.name,
                    "teacher_scoring": args.teacher_scoring,
                    "seed": args.seed,
                }
            )
            rows.append(row)
            if len(rows) >= args.chunk_size:
                rows_written += writer.write(rows)
                rows.clear()
                if writer.partitioned:
                    _write_checkpoint(output, game_number + 1, rows_written)
            state = environment.step(decision.letter)
        if args.log_every and (game_number + 1) % args.log_every == 0:
            logging.info("Generated %s/%s games and %s rows", game_number + 1, args.games, rows_written + len(rows))

    rows_written += writer.write(rows)
    writer.close()
    if writer.partitioned:
        _write_checkpoint(output, args.games, rows_written)
    logging.info("Wrote %s new states to %s", rows_written, output)


if __name__ == "__main__":
    main()
