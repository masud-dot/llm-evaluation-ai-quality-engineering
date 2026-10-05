from pathlib import Path

import pytest

from aiqe.changes import Change, detect
from aiqe.manifest import build_manifest, fingerprint

FIELDS = ["system_model", "provider_sdk", "retriever", "corpus",
          "tools", "judge_model", "judge_template", "rubric",
          "dataset", "sandbox"]


def config(root: Path, fields: list[str]) -> Path:
    body = 'prompt = { files = ["p/*.txt"] }\n' + "".join(
        f'{k} = {{ value = "x" }}\n' for k in fields)
    (root / "m.toml").write_text(body)
    return root / "m.toml"


def test_prompt_edit_is_detected(tmp_path: Path) -> None:
    (tmp_path / "p").mkdir()
    (tmp_path / "p/a.txt").write_text("one")
    cfg = config(tmp_path, FIELDS)
    before = build_manifest(tmp_path, cfg)
    assert build_manifest(tmp_path, cfg) == before
    (tmp_path / "p/a.txt").write_text("two")
    assert detect(before, build_manifest(tmp_path, cfg)) == {
        Change.PROMPT}


def test_incomplete_or_empty_config_fails(tmp_path: Path) -> None:
    (tmp_path / "p").mkdir()
    (tmp_path / "p/a.txt").write_text("one")
    with pytest.raises(ValueError):
        build_manifest(tmp_path, config(tmp_path, FIELDS[1:]))
    with pytest.raises(FileNotFoundError):
        fingerprint(tmp_path, ["nothing/*.txt"])
