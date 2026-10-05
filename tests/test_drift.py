import pytest

from aiqe.datasets import Reference, ReferenceKind
from aiqe.drift import input_drift, judge_drift, quality_shift
from aiqe.metrics import Rate
from aiqe.triage import Status, TriageItem, promote
from aiqe.validation import Confusion

BASE = {"tracking": 600, "claim": 200, "credit": 100,
        "billing": 80, "other": 20}


def test_no_drift_on_same_mix() -> None:
    d = input_drift(BASE, BASE)
    assert d.psi == pytest.approx(0.0, abs=1e-9)
    assert d.p_value > 0.99 and not d.growing


def test_new_topic_shows_as_growth() -> None:
    now = dict(BASE, other=140, bulky_delivery=60)
    d = input_drift(BASE, now)
    assert d.p_value < 0.001
    assert "bulky_delivery" in d.new_categories
    assert [g[0] for g in d.growing] == ["bulky_delivery",
                                         "other"]
    assert d.psi > 0.25


def test_quality_and_judge_drift() -> None:
    before = Rate(evaluator="e", passed=190, failed=10,
                  deferred=0)
    after = Rate(evaluator="e", passed=170, failed=30,
                 deferred=0)
    assert quality_shift(before, after) < 0.01
    v = Confusion(tp=57, fn=3, fp=12, tn=18, deferred=0)
    r = Confusion(tp=40, fn=2, fp=14, tn=6, deferred=0)
    delta, p = judge_drift(v, r)
    assert delta < 0 and 0 < p < 1


def test_triage_to_regression_case() -> None:
    item = TriageItem(item_id="0042", conversation_id="conv-9",
                      signal="pre-filter fail")
    with pytest.raises(ValueError):
        item.move(Status.CONFIRMED, "lead", "no mode yet")
    item = item.model_copy(update={
        "failure_mode": "FM-CREDIT-PROMISE",
        "objective_ids": ["OBJ-CREDIT-01"]})
    item = item.move(Status.CONFIRMED, "credit-lead", "real")
    ref = Reference(kind=ReferenceKind.CRITERIA,
                    values=["No promise of compensation"])
    with pytest.raises(ValueError):
        promote(item, "Call me on 07700 900123, sofa late",
                ref)
    done, case = promote(item, "My sofa is two days late. "
                         "Credit?", ref)
    assert done.status is Status.PROMOTED
    assert case.case_id == "reg-0042"
    assert "fm:FM-CREDIT-PROMISE" in case.tags
    with pytest.raises(ValueError):
        done.move(Status.NEW, "x", "reopen")
