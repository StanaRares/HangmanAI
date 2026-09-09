from __future__ import annotations

import json

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app, get_service
from app.tree.builder import HangmanTreeBuilder, TreeBuildConfig
from app.tree.evaluation import evaluate_tree
from app.tree.serialization import save_tree

CHECKPOINT_WORDS = (
    "babba",
    "bacca",
    "badda",
    "baffa",
    "bagga",
    "bahha",
    "bakka",
    "balla",
    "bamma",
    "banna",
    "bappa",
    "barra",
    "bassa",
    "batta",
)


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


def build_budget_checkpoint_client(tmp_path, monkeypatch) -> tuple[TestClient, int]:
    words = [*CHECKPOINT_WORDS, "cabal", "sissy"]
    frame = pd.DataFrame(
        {
            "word": words,
            "length": [len(word) for word in words],
            "zipf_frequency": [3.0 for _ in words],
        }
    )
    vocabulary_path = tmp_path / "vocabulary.parquet"
    frame.to_parquet(vocabulary_path, index=False)

    config = TreeBuildConfig.from_profile(length=5, profile="fast", node_budget=4)
    tree = HangmanTreeBuilder(words, config=config).build(forced_root_letter="a")
    budget_node = next(node for node in tree.nodes.values() if node.pattern == "_a__a")
    model_path = tmp_path / "models" / "length_5" / "uniform" / "tree.json.gz"
    save_tree(tree, model_path)

    monkeypatch.setenv("HANGMAN_VOCABULARY", str(vocabulary_path))
    monkeypatch.setenv("HANGMAN_MODELS_DIR", str(tmp_path / "models"))
    get_service.cache_clear()
    return TestClient(app), budget_node.node_id


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


def test_budget_node_payload_is_expandable_checkpoint(tmp_path, monkeypatch) -> None:
    client, budget_node_id = build_budget_checkpoint_client(tmp_path, monkeypatch)

    response = client.get(f"/trees/5/node/{budget_node_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["guess"] is None
    assert payload["leaf_type"] == "budget"
    assert payload["expandable"] is True
    assert payload["candidate_count"] == len(CHECKPOINT_WORDS)
    assert payload["reconstructed_candidate_count"] == len(CHECKPOINT_WORDS)
    assert payload["candidate_count_mismatch"] is False
    assert payload["candidate_sample"] == list(CHECKPOINT_WORDS[:12])


def test_candidate_reconstruction_uses_full_vocabulary_and_repeated_constraints(tmp_path, monkeypatch) -> None:
    _, budget_node_id = build_budget_checkpoint_client(tmp_path, monkeypatch)
    service = get_service()
    node = service.load_tree(5).get_node(budget_node_id)

    candidates = service.reconstruct_node_candidates(5, node)

    assert set(candidates) == set(CHECKPOINT_WORDS)
    assert len(candidates) > len(node.candidate_sample)
    assert "cabal" not in candidates


def test_expand_budget_node_persists_and_traverses_after_reload(tmp_path, monkeypatch) -> None:
    client, budget_node_id = build_budget_checkpoint_client(tmp_path, monkeypatch)
    before = client.get(f"/trees/5/node/{budget_node_id}").json()

    response = client.post(
        f"/trees/5/node/{budget_node_id}/expand",
        json={"node_budget": 80},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["expanded"] is True
    assert payload["already_expanded"] is False
    assert payload["node_id"] == budget_node_id
    assert payload["pattern"] == before["pattern"]
    assert payload["guessed_letters"] == before["guessed_letters"]
    assert payload["incorrect_letters"] == before["incorrect_letters"]
    assert payload["remaining_lives"] == before["remaining_lives"]
    assert payload["candidate_count"] == before["candidate_count"]
    assert payload["guess"] is not None
    assert payload["guess"] != "a"
    assert payload["leaf_type"] is None
    assert payload["branches"]

    branch = payload["branches"][0]
    branch_response = client.get(f"/trees/5/node/{branch['node_id']}")
    assert branch_response.status_code == 200
    branch_payload = branch_response.json()
    assert branch_payload["node_id"] == branch["node_id"]
    assert branch_payload["pattern"] == branch["resulting_pattern"]

    extension_path = tmp_path / "models" / "length_5" / "uniform" / "extensions" / f"node_{budget_node_id}.json.gz"
    assert extension_path.exists()

    repeated = client.post(
        f"/trees/5/node/{budget_node_id}/expand",
        json={"node_budget": 80},
    )
    assert repeated.status_code == 200
    assert repeated.json()["already_expanded"] is True
    assert repeated.json()["guess"] == payload["guess"]

    get_service.cache_clear()
    reloaded_client = TestClient(app)
    reloaded = reloaded_client.get(f"/trees/5/node/{budget_node_id}")
    assert reloaded.status_code == 200
    reloaded_payload = reloaded.json()
    assert reloaded_payload["guess"] == payload["guess"]
    assert reloaded_payload["extension_loaded"] is True
    assert reloaded_payload["branches"]
