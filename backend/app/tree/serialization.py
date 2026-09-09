from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.tree.model import HangmanDecisionTree


def _tree_bytes(tree: HangmanDecisionTree) -> bytes:
    data = json.dumps(tree.to_dict(), separators=(",", ":")).encode("utf-8")
    return data


def _write_tree_bytes(destination: Path, data: bytes, compress: bool) -> None:
    if compress:
        with gzip.open(destination, "wb") as handle:
            handle.write(data)
    else:
        destination.write_bytes(data)


def save_tree(tree: HangmanDecisionTree, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_tree_bytes(destination, _tree_bytes(tree), destination.suffix == ".gz")
    return destination


def save_tree_atomic(tree: HangmanDecisionTree, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        _write_tree_bytes(temporary, _tree_bytes(tree), destination.suffix == ".gz")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_tree(path: str | Path) -> HangmanDecisionTree:
    source = Path(path)
    if source.suffix == ".gz":
        with gzip.open(source, "rb") as handle:
            payload: dict[str, Any] = json.loads(handle.read().decode("utf-8"))
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
    return HangmanDecisionTree.from_dict(payload)
