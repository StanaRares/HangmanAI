from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from app.tree.model import HangmanDecisionTree


def save_tree(tree: HangmanDecisionTree, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(tree.to_dict(), separators=(",", ":")).encode("utf-8")
    if destination.suffix == ".gz":
        with gzip.open(destination, "wb") as handle:
            handle.write(data)
    else:
        destination.write_bytes(data)
    return destination


def load_tree(path: str | Path) -> HangmanDecisionTree:
    source = Path(path)
    if source.suffix == ".gz":
        with gzip.open(source, "rb") as handle:
            payload: dict[str, Any] = json.loads(handle.read().decode("utf-8"))
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
    return HangmanDecisionTree.from_dict(payload)
