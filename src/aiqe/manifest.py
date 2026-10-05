"""aiqe/manifest.py: fingerprint the components of a system."""
from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

from pydantic import BaseModel

from aiqe.changes import Manifest


class Component(BaseModel):
    """Either files to hash or a literal identifier."""

    files: list[str] = []      # glob patterns, repo-relative
    value: str | None = None   # e.g. a dated model snapshot


def fingerprint(root: Path, patterns: list[str]) -> str:
    """Content hash over matched files, independent of order."""
    paths = sorted({p for pat in patterns
                    for p in root.glob(pat) if p.is_file()})
    if not paths:
        raise FileNotFoundError(f"no files match {patterns}")
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def build_manifest(root: Path, config: Path) -> Manifest:
    with config.open("rb") as fh:
        spec = {k: Component.model_validate(v)
                for k, v in tomllib.load(fh).items()}
    missing = set(Manifest.model_fields) - set(spec)
    if missing:
        raise ValueError(f"manifest config lacks {sorted(missing)}")
    values = {}
    for name, comp in spec.items():
        if comp.value is not None:
            values[name] = comp.value
        else:
            values[name] = fingerprint(root, comp.files)
    return Manifest.model_validate(values)
