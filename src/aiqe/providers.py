"""aiqe/providers.py (excerpt): the vendor-neutral seam."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel


class Completion(BaseModel):
    """What any provider returns to the rest of aiqe."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int


class Provider(Protocol):
    def complete(self, model: str, prompt: str) -> Completion:
        ...


def replay_key(model: str, prompt: str) -> str:
    """Stable key for a (model, prompt) pair."""
    raw = json.dumps([model, prompt]).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


class ReplayProvider:
    """Serves recorded completions; fails loudly on a miss."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def complete(self, model: str, prompt: str) -> Completion:
        path = self.root / f"{replay_key(model, prompt)}.json"
        if not path.exists():
            raise LookupError(f"no replay for {path.name}")
        return Completion.model_validate_json(path.read_text())


class RecordingProvider:
    """Wraps a live provider and records every completion."""

    def __init__(self, inner: Provider, root: Path) -> None:
        self.inner = inner
        self.root = root

    def complete(self, model: str, prompt: str) -> Completion:
        out = self.inner.complete(model, prompt)
        path = self.root / f"{replay_key(model, prompt)}.json"
        path.write_text(out.model_dump_json())
        return out


def replay_root(base: Path, sampling: dict[str, float]) -> Path:
    """Recordings live under a folder named for the sampling
    settings, so changing them causes a loud replay miss."""
    tag = json.dumps(sorted(sampling.items()))
    return base / hashlib.sha256(tag.encode()).hexdigest()[:12]
