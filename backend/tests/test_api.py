from __future__ import annotations

import json

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app, get_service
from app.tree.builder import HangmanTreeBuilder, TreeBuildConfig
from app.tree.evaluation import evaluate_tree
from app.tree.serialization import save_tree


def build_test_client(tmp_path, monkeypatch) -> TestClient:
    words = ["cat", "car", "can", "dog"]
    frame = pd.DataFrame(
        {
            "word": words,
            "length": [len(word) for word in words],
            "zipf_frequency": [4.0, 3.0, 3.0, 2.0],
        }
    )
    vocabulary_path = tmp_path / "vocabulary.parquet"
    frame.to_parquet(vocabulary_path, index=False)

    config = TreeBuildConfig.from_profile(length=3, profile="fast", node_budget=5000)
    tree = HangmanTreeBuilder(words, config=config).build()
    model_dir = tmp_path / "models" / "length_3" / "uniform"
    model_path = model_dir / "tree.json.gz"
    save_tree(tree, model_path)
    summary, _ = evaluate_tree(tree, words, model_path)
    (model_dir / "metadata.json").write_text(json.dumps(tree.metadata), encoding="utf-8")
    (model_dir / "vocabulary_stats.json").write_text(
        json.dumps({"length": 3, "words": len(words), "weighting": "uniform"}),
        encoding="utf-8",
    )
    (model_dir / "evaluation_results.json").write_text(json.dumps(summary), encoding="utf-8")
    (model_dir / "training_statistics.json").write_text(
        json.dumps(
            {
                "length": 3,
                "words": len(words),
                "best_first_guess": tree.best_first_guess,
                "node_count": tree.node_count,
                "leaf_count": tree.leaf_count,
                "max_depth": tree.max_depth,
                "strategy": tree.strategy,
                "weighting": "uniform",
                "training_time": tree.metadata["training_seconds"],
                **summary,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("HANGMAN_VOCABULARY", str(vocabulary_path))
    monkeypatch.setenv("HANGMAN_MODELS_DIR", str(tmp_path / "models"))
    get_service.cache_clear()
    return TestClient(app)


def test_health_and_vocabulary_endpoints(tmp_path, monkeypatch) -> None:
    client = build_test_client(tmp_path, monkeypatch)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["vocabulary_size"] == 4
    assert response.json()["available_tree_lengths"] == [3]

    response = client.get("/vocabulary/summary")
    assert response.status_code == 200
    assert response.json()[0]["length"] == 3


def test_tree_endpoints_return_lazy_node_payloads(tmp_path, monkeypatch) -> None:
    client = build_test_client(tmp_path, monkeypatch)

    response = client.get("/trees")
    assert response.status_code == 200
    assert response.json()["trees"][0]["length"] == 3

    response = client.get("/trees/3/node/0")
    assert response.status_code == 200
    payload = response.json()
    assert payload["node_id"] == 0
    assert payload["guess"] is not None
    assert "branches" in payload


def test_tree_game_lifecycle(tmp_path, monkeypatch) -> None:
    client = build_test_client(tmp_path, monkeypatch)

    response = client.post("/game/new", json={"word": "cat", "max_lives": 6})
    assert response.status_code == 200
    game_id = response.json()["game_id"]
    assert "solution" not in response.json()
    assert response.json()["model"]["length"] == 3

    response = client.post(f"/game/{game_id}/tree-step")
    assert response.status_code == 200
    assert "decision" in response.json()
    assert response.json()["decision"]["node_id"] == 0

    response = client.post(f"/game/{game_id}/tree-play", json={"max_steps": 26})
    assert response.status_code == 200
    assert response.json()["state"]["status"] in {"won", "lost"}
    assert response.json()["solution"] == "cat"
