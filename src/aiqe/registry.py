"""aiqe/registry.py: an append-only record of dataset versions."""
from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel

from aiqe.datasets import Dataset


class Entry(BaseModel):
    name: str
    version: str
    content_hash: str
    path: str


class Registry:
    """A name and version always mean the same content."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def entries(self) -> list[Entry]:
        if not self.path.exists():
            return []
        with self.path.open("rb") as fh:
            data = tomllib.load(fh)
        return [Entry.model_validate(e)
                for e in data.get("datasets", [])]

    def register(self, ds: Dataset, path: str) -> Entry:
        new = Entry(name=ds.name, version=ds.version,
                    content_hash=ds.content_hash(), path=path)
        for e in self.entries():
            if (e.name, e.version) == (new.name, new.version):
                if e.content_hash != new.content_hash:
                    raise ValueError(
                        f"{e.name} {e.version} already means "
                        "different content; bump the version")
                return e
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write("\n[[datasets]]\n"
                     f'name = "{new.name}"\n'
                     f'version = "{new.version}"\n'
                     f'content_hash = "{new.content_hash}"\n'
                     f'path = "{new.path}"\n')
        return new

    def latest(self, name: str) -> Entry:
        matches = [e for e in self.entries() if e.name == name]
        if not matches:
            raise KeyError(name)
        return matches[-1]
