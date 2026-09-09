from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.services.tree_service import ExpansionInProgressError, TreeConsistencyError, TreeGameService


class NewGameRequest(BaseModel):
    word: str | None = None
    length: int | None = Field(default=None, ge=3)
    max_lives: int = Field(default=6, ge=1, le=12)
    weighting: str = "uniform"


class GuessRequest(BaseModel):
    letter: str


class TreePlayRequest(BaseModel):
    max_steps: int = Field(default=26, ge=1, le=26)


class ExpandNodeRequest(BaseModel):
    node_budget: int = Field(default=20000, ge=2, le=200000)
    profile: str = "fast"


class AiGuessRequest(BaseModel):
    agent: str = "hangman_decision_tree"


class SimulateRequest(BaseModel):
    games: int = Field(default=1, ge=1, le=1000)
    word: str | None = None
    length: int | None = Field(default=None, ge=3)
    max_lives: int = Field(default=6, ge=1, le=12)
    weighting: str = "uniform"


@lru_cache
def get_service() -> TreeGameService:
    return TreeGameService(
        vocabulary_path=os.getenv("HANGMAN_VOCABULARY"),
        models_dir=os.getenv("HANGMAN_MODELS_DIR"),
    )


app = FastAPI(
    title="HangmanAI Decision Tree API",
    description="Per-word-length multiway decision trees for playing Hangman.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _game_payload(
    service: TreeGameService,
    game_id: str,
    decision: dict[str, Any] | None = None,
    decisions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    session = service.get_game(game_id)
    state = session.environment.get_state()
    payload: dict[str, Any] = {
        "game_id": game_id,
        "state": state.to_public_dict(),
        "analysis": service.candidate_analysis(state),
        "model": {
            "length": session.tree.length,
            "weighting": session.weighting,
            "model_source": session.model_source,
            "current_node_id": session.node_id,
            "tree_position_valid": session.tree_position_valid,
            "in_vocabulary": session.in_vocabulary,
            "best_first_guess": session.tree.best_first_guess,
            "node_count": session.tree.node_count,
            "max_depth": session.tree.max_depth,
            "training_strategy": session.tree.strategy,
        },
        "decisions": session.decisions,
    }
    if decision:
        payload["decision"] = decision
    if decisions is not None:
        payload["new_decisions"] = decisions
    if state.is_finished:
        payload["solution"] = session.environment.solution
    return payload


@app.get("/health")
def health() -> dict[str, Any]:
    return get_service().health()


@app.get("/vocabulary/summary")
def get_vocabulary_summary() -> list[dict[str, Any]]:
    return get_service().vocabulary_summary()


@app.get("/trees")
def list_trees() -> dict[str, Any]:
    return {"trees": get_service().list_trees()}


@app.get("/trees/{length}")
def get_tree(length: int, weighting: str = "uniform") -> dict[str, Any]:
    try:
        return get_service().tree_summary(length, weighting)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/trees/{length}/summary")
def get_tree_summary(length: int, weighting: str = "uniform") -> dict[str, Any]:
    return get_tree(length, weighting)


@app.get("/trees/{length}/root-analysis")
def get_root_analysis(length: int, weighting: str = "uniform") -> list[dict[str, Any]]:
    return get_service().root_analysis(length, weighting)


@app.get("/trees/{length}/node/{node_id}")
def get_tree_node(
    length: int,
    node_id: int,
    weighting: str = "uniform",
    materialize: bool = False,
    node_budget: int = 20000,
) -> dict[str, Any]:
    try:
        if materialize:
            return get_service().materialized_node_payload(length, node_id, weighting, node_budget)
        return get_service().node_payload(length, node_id, weighting)
    except ExpansionInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TreeConsistencyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/trees/{length}/node/{node_id}/expand")
def expand_tree_node(
    length: int,
    node_id: int,
    request: ExpandNodeRequest = ExpandNodeRequest(),
    weighting: str = "uniform",
) -> dict[str, Any]:
    try:
        return get_service().expand_node(
            length=length,
            node_id=node_id,
            weighting=weighting,
            node_budget=request.node_budget,
            profile=request.profile,
        )
    except ExpansionInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TreeConsistencyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/game/new")
def new_game(request: NewGameRequest) -> dict[str, Any]:
    try:
        service = get_service()
        game_id, _ = service.create_game(
            word=request.word,
            length=request.length,
            max_lives=request.max_lives,
            weighting=request.weighting,
        )
        return _game_payload(service, game_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/game/{game_id}")
def get_game(game_id: str) -> dict[str, Any]:
    try:
        service = get_service()
        return _game_payload(service, game_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/game/{game_id}/guess")
def guess(game_id: str, request: GuessRequest) -> dict[str, Any]:
    try:
        service = get_service()
        _, decision = service.manual_guess(game_id, request.letter)
        return _game_payload(service, game_id, decision=decision)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TreeConsistencyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/game/{game_id}/tree-step")
def tree_step(game_id: str) -> dict[str, Any]:
    try:
        service = get_service()
        _, decision = service.tree_step(game_id)
        return _game_payload(service, game_id, decision=decision)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TreeConsistencyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/game/{game_id}/tree-play")
def tree_play(game_id: str, request: TreePlayRequest = TreePlayRequest()) -> dict[str, Any]:
    try:
        service = get_service()
        _, decisions = service.tree_play(game_id, max_steps=request.max_steps)
        return _game_payload(service, game_id, decisions=decisions)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TreeConsistencyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/results/by-length")
def results_by_length() -> list[dict[str, Any]]:
    return get_service().results_by_length()


@app.get("/results/hardest-words/{length}")
def hardest_words(length: int) -> list[dict[str, Any]]:
    return get_service().hardest_words(length)


@app.get("/stats")
def stats() -> dict[str, Any]:
    service = get_service()
    return {**service.health(), "performance_by_length": service.results_by_length()}


@app.get("/agents")
def agents() -> dict[str, Any]:
    trees = get_service().list_trees()
    return {
        "agents": [
            {
                "name": "hangman_decision_tree",
                "trained": bool(trees),
                "description": "The runtime player traverses the serialized multiway tree for the hidden word length.",
            }
        ],
        "legacy_note": "Older random, entropy, forest, and boosting agents remain in code for history but are not the main product.",
    }


@app.post("/game/{game_id}/ai-guess")
def ai_guess_legacy_alias(game_id: str, request: AiGuessRequest) -> dict[str, Any]:
    if request.agent != "hangman_decision_tree":
        raise HTTPException(
            status_code=400,
            detail="The public API now exposes only the per-length Hangman decision tree player.",
        )
    return tree_step(game_id)


@app.post("/simulate")
def simulate(request: SimulateRequest) -> dict[str, Any]:
    service = get_service()
    results = []
    for _ in range(request.games):
        try:
            game_id, _ = service.create_game(
                word=request.word,
                length=request.length,
                max_lives=request.max_lives,
                weighting=request.weighting,
            )
            state, decisions = service.tree_play(game_id)
            session = service.get_game(game_id)
            results.append(
                {
                    "word": session.environment.solution,
                    "status": state.status,
                    "turns": state.turn,
                    "incorrect_guesses": len(state.incorrect_letters),
                    "remaining_lives": state.remaining_lives,
                    "history": [record.to_dict() for record in state.history],
                    "decisions": decisions,
                }
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except TreeConsistencyError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    wins = sum(1 for result in results if result["status"] == "won")
    return {
        "agent": "hangman_decision_tree",
        "games": len(results),
        "win_rate": wins / len(results) if results else 0.0,
        "results": results,
    }
