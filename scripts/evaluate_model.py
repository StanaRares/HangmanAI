from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.agents.decision_tree_agent import DecisionTreeAgent  # noqa: E402
from app.core.candidate_filter import CandidateIndex  # noqa: E402
from app.core.dictionary import PROJECT_ROOT, read_words  # noqa: E402
from app.core.environment import HangmanEnvironment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DecisionTreeAgent.")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models" / "decision_tree.joblib"))
    parser.add_argument("--dictionary", default=str(PROJECT_ROOT / "data" / "processed" / "train.txt"))
    parser.add_argument("--test-words", default=str(PROJECT_ROOT / "data" / "processed" / "test.txt"))
    parser.add_argument("--max-games", type=int, default=500)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports" / "results" / "decision_tree_evaluation.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dictionary = read_words(args.dictionary)
    test_words = read_words(args.test_words)[: args.max_games]
    index = CandidateIndex(dictionary)
    agent = DecisionTreeAgent(args.model)
    wins = 0
    wrong = []
    for word in test_words:
        environment = HangmanEnvironment(dictionary or [word], max_lives=6)
        state = environment.reset(word)
        while not state.is_finished:
            decision = agent.guess(state, index)
            state = environment.step(decision.letter)
        wins += int(state.status == "won")
        wrong.append(len(state.incorrect_letters))
    result = {
        "agent": agent.name,
        "model_loaded": agent.is_trained,
        "games": len(test_words),
        "win_rate": wins / len(test_words) if test_words else 0.0,
        "average_wrong_guesses": sum(wrong) / len(wrong) if wrong else 0.0,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logging.info("Wrote %s", output)


if __name__ == "__main__":
    main()
