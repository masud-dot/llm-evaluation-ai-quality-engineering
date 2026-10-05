from pathlib import Path

import pytest
from pydantic import ValidationError

from aiqe.datasets import Split, load_jsonl
from aiqe.evaluators import Score
from aiqe.metrics import pass_rate, sliced
from aiqe.rubrics import Criterion, Level, Rubric, Scale
from aiqe.runner import RunResult


def sc(cid: str, p: bool | None) -> Score:
    return Score(case_id=cid, evaluator="e", passed=p)


def test_rate_keeps_denominator_and_coverage() -> None:
    r = pass_rate([sc("a", True), sc("b", False),
                   sc("c", None), sc("d", True)], "e")
    assert (r.passed, r.failed, r.deferred) == (2, 1, 1)
    assert r.value == pytest.approx(2 / 3)
    assert r.coverage == pytest.approx(3 / 4)
    empty = pass_rate([sc("a", None)], "e")
    assert empty.value is None and empty.coverage == 0.0


def test_sliced_by_tag() -> None:
    ds = load_jsonl(
        Path("datasets/compass/credit-claims-v1.jsonl"),
        "credit-claims", "1.0")
    run = RunResult(
        dataset=ds.name, dataset_version=ds.version,
        dataset_hash=ds.content_hash(), split=Split.HOLDOUT,
        system="stub", outputs={},
        scores=[sc("cc-001", True), sc("cc-002", False),
                sc("cc-003", True)])
    s = sliced(run, ds, "e", ["priority", "standard", "claim"])
    assert s["priority"].value == 1.0
    assert s["standard"].value == 0.0
    assert s["claim"].decided == 1


def binary(levels: list[int]) -> Rubric:
    return Rubric(id="r", version="1", objective_id="OBJ-X-01",
                  scale=Scale.BINARY, criteria=[Criterion(
                      id="c", question="q",
                      levels=[Level(value=v, anchor="a")
                              for v in levels])])


def test_rubric_validation_and_render() -> None:
    r = binary([0, 1])
    assert "0: a" in r.render()
    with pytest.raises(ValidationError):
        binary([1, 2])
    with pytest.raises(ValidationError):
        Rubric(id="r", version="1", objective_id="O",
               scale=Scale.GRADED, criteria=[Criterion(
                   id="c", question="q",
                   levels=[Level(value=0, anchor="a"),
                           Level(value=2, anchor="b")])])


def test_credit_rubric_file_loads() -> None:
    import tomllib
    path = Path("rubrics/credit-commitment.toml")
    with path.open("rb") as fh:
        r = Rubric.model_validate(tomllib.load(fh))
    assert r.scale is Scale.BINARY
    assert all(len(x) <= 70 for x in r.render().splitlines())
