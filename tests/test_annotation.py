import csv
import tomllib
from pathlib import Path

import pytest

from aiqe.annotation import (Adjudication, HumanLabel, alpha,
                             as_scores, export_batch,
                             final_labels, import_labels, kappa,
                             output_hash, percent_agreement)
from aiqe.datasets import Split, load_jsonl
from aiqe.rubrics import Rubric
from aiqe.runner import run


def rubric() -> Rubric:
    with Path("rubrics/credit-commitment.toml").open("rb") as f:
        return Rubric.model_validate(tomllib.load(f))


def lab(i: int, ann: str, v: int) -> HumanLabel:
    return HumanLabel(case_id=f"c{i}", output_hash="h",
                      rubric_id="r", rubric_version="1",
                      criterion_id="no-guarantee",
                      annotator=ann, value=v)


def constructed() -> list[HumanLabel]:
    """50 items: 35 both pass, 5 both fail, 6 A-only, 4 B-only."""
    pat = ([(1, 1)] * 35 + [(0, 0)] * 5 + [(1, 0)] * 6
           + [(0, 1)] * 4)
    out = []
    for i, (a, b) in enumerate(pat):
        out += [lab(i, "A", a), lab(i, "B", b)]
    return out


def test_agreement_on_constructed_example() -> None:
    lbs = constructed()
    assert percent_agreement(lbs, "A", "B") == pytest.approx(0.8)
    k = kappa(lbs, "A", "B", n_levels=2)
    assert k == pytest.approx(0.3766, abs=1e-3)
    assert -1 <= alpha(lbs, ["A", "B"]) <= 1


def test_export_import_roundtrip(tmp_path: Path) -> None:
    ds = load_jsonl(
        Path("datasets/compass/credit-claims-v1.jsonl"),
        "credit-claims", "1.0")
    res = run(ds, Split.HOLDOUT, "stub",
              lambda q: "A credit may apply after review.", [])
    path = tmp_path / "batch.csv"
    export_batch(res, ds, rubric(), path)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 3
    for r in rows:
        r["value"] = "1"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    labels = import_labels(path, "A", rubric())
    assert {lb.output_hash for lb in labels} == {
        output_hash("A credit may apply after review.")}


def test_bad_level_rejected(tmp_path: Path) -> None:
    p = tmp_path / "b.csv"
    p.write_text("case_id,output_hash,input,output,"
                 "criterion_id,value,note\n"
                 "c1,h,q,o,no-guarantee,3,\n")
    with pytest.raises(ValueError):
        import_labels(p, "A", rubric())


def test_adjudication_and_scores() -> None:
    lbs = [lab(1, "A", 1), lab(1, "B", 1),
           lab(2, "A", 1), lab(2, "B", 0)]
    with pytest.raises(ValueError):
        final_labels(lbs, [])
    adj = Adjudication(case_id="c2", output_hash="h",
                       criterion_id="no-guarantee", value=0,
                       adjudicator="lead", rationale="implied")
    final = final_labels(lbs, [adj])
    scores = as_scores(final, rubric())
    assert {s.case_id: s.passed for s in scores} == {
        "c1": True, "c2": False}
