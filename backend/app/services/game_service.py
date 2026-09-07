from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from uuid import uuid4

from app.agents import (
    CandidateFrequencyAgent,
    BoostingAgent,
    DecisionTreeAgent,
    EntropyAgent,
    GlobalFrequencyAgent,
    RandomAgent,
    RandomForestAgent,
)
from app.agents.base import AgentDecision
from app.core.candidate_filter import CandidateIndex
from app.core.dictionary import PROJECT_ROOT, load_default_words, read_words
from app.core.environment import HangmanEnvironment
from app.core.state import GameState


class GameService:
    def __init__(
        self,
        dictionary_path: str | Path | None = None,
        model_path: str | Path | None = None,
        seed: int = 42,
    ) -> None:
        if dictionary_path:
            self.words = read_words(dictionary_path)
        else:
            self.words = load_default_words()
        self.index = CandidateIndex(self.words)
        self.seed = seed
        self.rng = random.Random(seed)
        default_model = PROJECT_ROOT / "models" / "decision_tree.joblib"
        self.model_path = Path(model_path) if model_path else default_model
        self.agents = {
            "random": RandomAgent(seed=seed),
            "global_frequency": GlobalFrequencyAgent(self.words),
            "candidate_frequency": CandidateFrequencyAgent(),
            "entropy": EntropyAgent(scoring="entropy_only", name="entropy"),
            "risk_adjusted_entropy": EntropyAgent(scoring="risk_adjusted_entropy"),
            "decision_tree": DecisionTreeAgent(self.model_path),
            "random_forest": RandomForestAgent(PROJECT_ROOT / "models" / "random_forest.joblib"),
            "boosting": BoostingAgent(PROJECT_ROOT / "models" / "boosting.joblib"),
        }
        self.games: dict[str, HangmanEnvironment] = {}

    def create_game(self, word: str | None = None, max_lives: int = 6) -> tuple[str, GameState]:
        game_id = str(uuid4())
        environment = HangmanEnvironment(self.words, max_lives=max_lives, seed=self.rng.randint(0, 10**9))
        state = environment.reset(word)
        self.games[game_id] = environment
        return game_id, state

    def get_game(self, game_id: str) -> HangmanEnvironment:
        if game_id not in self.games:
            raise KeyError(f"Unknown game_id: {game_id}")
        return self.games[game_id]

    def manual_guess(self, game_id: str, letter: str) -> GameState:
        return self.get_game(game_id).step(letter)

    def ai_guess(self, game_id: str, agent_name: str) -> tuple[GameState, AgentDecision]:
        environment = self.get_game(game_id)
        agent = self.get_agent(agent_name)
        decision = agent.guess(environment.get_state(), self.index)
        state = environment.step(decision.letter)
        return state, decision

    def get_agent(self, name: str):
        if name not in self.agents:
            raise KeyError(f"Unknown agent: {name}")
        return self.agents[name]

    def list_agents(self) -> list[dict[str, object]]:
        return [
            {
                "name": name,
                "trained": getattr(agent, "is_trained", True),
                "description": self._agent_description(name),
            }
            for name, agent in self.agents.items()
        ]

    def simulate(
        self,
        agent_name: str,
        games: int = 1,
        word: str | None = None,
        max_lives: int = 6,
    ) -> dict[str, object]:
        agent = self.get_agent(agent_name)
        words = [word] if word else [self.rng.choice(self.words) for _ in range(games)]
        results = []
        for selected_word in words:
            environment = HangmanEnvironment(self.words, max_lives=max_lives)
            state = environment.reset(selected_word)
            decisions = []
            while not state.is_finished:
                started = time.perf_counter()
                decision = agent.guess(state, self.index)
                elapsed_ms = (time.perf_counter() - started) * 1000
                state = environment.step(decision.letter)
                decisions.append({**decision.to_dict(), "elapsed_ms": round(elapsed_ms, 4)})
            results.append(
                {
                    "word": selected_word if state.is_finished else None,
                    "status": state.status,
                    "turns": state.turn,
                    "incorrect_guesses": len(state.incorrect_letters),
                    "remaining_lives": state.remaining_lives,
                    "history": [record.to_dict() for record in state.history],
                    "decisions": decisions,
                }
            )
        win_rate = sum(1 for result in results if result["status"] == "won") / len(results)
        return {"agent": agent_name, "games": len(results), "win_rate": win_rate, "results": results}

    def stats(self) -> dict[str, object]:
        latest_summary = PROJECT_ROOT / "reports" / "results" / "model_generalization" / "benchmark_summary.csv"
        if not latest_summary.exists():
            latest_summary = PROJECT_ROOT / "reports" / "results" / "benchmark_summary.csv"
        benchmark_rows: list[dict[str, str]] = []
        if latest_summary.exists():
            with latest_summary.open("r", encoding="utf-8", newline="") as handle:
                benchmark_rows = list(csv.DictReader(handle))
        return {
            "dictionary_size": len(self.words),
            "word_lengths": sorted({len(word) for word in self.words}),
            "decision_tree_model_loaded": getattr(self.agents["decision_tree"], "is_trained", False),
            "benchmark_summary": benchmark_rows,
        }

    @staticmethod
    def _agent_description(name: str) -> str:
        descriptions = {
            "random": "Uniform random choice among unguessed letters.",
            "global_frequency": "Ranks letters by document frequency in the training dictionary.",
            "candidate_frequency": "Filters compatible words and ranks letters by candidate document frequency.",
            "entropy": "Chooses the letter with maximum expected information gain.",
            "risk_adjusted_entropy": "Splits candidates while penalizing risky misses as lives get low.",
            "decision_tree": "Predicts the next letter from engineered state and candidate features.",
            "random_forest": "Tree ensemble baseline trained on the same supervised state data.",
            "boosting": "Gradient boosting baseline trained on the same supervised state data.",
        }
        return descriptions.get(name, "")
