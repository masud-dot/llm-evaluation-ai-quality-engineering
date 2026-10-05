"""The recording, labelling and figure scripts, run offline."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "scripts")
import record_fixtures  # noqa: E402


def test_dry_run_makes_no_calls(tmp_path: Path,
                                capsys: pytest.CaptureFixture[str]
                                ) -> None:
    rc = record_fixtures.main(["--provider", "stub", "--dry-run",
                               "--trials", "3", "--out",
                               str(tmp_path / "r")])
    out = capsys.readouterr().out
    assert rc == 0 and "no calls were made" in out
    assert "system calls: 12" in out      # 4 cases x 3 trials
    assert not (tmp_path / "r").exists()


def test_stub_recording_then_replay(tmp_path: Path) -> None:
    rc = record_fixtures.main([
        "--provider", "stub", "--trials", "2",
        "--out", str(tmp_path / "r"),
        "--judge-out", str(tmp_path / "j")])
    assert rc == 0
    from aiqe.providers import replay_root
    base = replay_root(tmp_path / "r", {})
    assert len(list((base / "trial-0").glob("*.json"))) == 4
    assert list((tmp_path / "j").glob("*.json"))


def test_figures_refuse_missing_input(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")   # the book-only figures extra
    import make_figures
    with pytest.raises(SystemExit):
        make_figures.main(["--confusion",
                           str(tmp_path / "nope.json"),
                           "--out", str(tmp_path)])


def test_figures_from_recorded_evidence(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")   # the book-only figures extra
    import make_figures
    (tmp_path / "c.json").write_text(json.dumps(
        {"tp": 57, "fn": 3, "fp": 12, "tn": 18, "deferred": 2}))
    (tmp_path / "t.json").write_text(json.dumps(
        {f"c{i}": [True, True, i % 5 != 0] for i in range(20)}))
    (tmp_path / "m.json").write_text(json.dumps([
        {"name": "current", "cost": 0.012, "groundedness": 0.86,
         "pass3_lower": 0.96},
        {"name": "smaller", "cost": 0.006, "groundedness": 0.85,
         "pass3_lower": 0.90},
        {"name": "larger", "cost": 0.031, "groundedness": 0.90,
         "pass3_lower": 0.97}]))
    assert make_figures.main([
        "--confusion", str(tmp_path / "c.json"),
        "--trials", str(tmp_path / "t.json"),
        "--candidates", str(tmp_path / "m.json"),
        "--out", str(tmp_path)]) == 0
    for n in ("F09", "F10", "F16"):
        assert (tmp_path / f"{n}.png").stat().st_size > 5000
