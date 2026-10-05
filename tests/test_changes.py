from aiqe.changes import (Change, Manifest, detect, impact,
                          regression_case)
from aiqe.datasets import Reference, ReferenceKind, Source, Split

BASE = Manifest(prompt="p1", system_model="m-2026-01",
                provider_sdk="openai==3.15.0", retriever="r1",
                corpus="c1", tools="t1", judge_model="j-2026-01",
                judge_template="jt1", rubric="1.0",
                dataset="d1", sandbox="s1")


def test_detect_and_route_model_migration() -> None:
    after = BASE.model_copy(update={"system_model": "m-2026-09"})
    changes = detect(BASE, after)
    assert changes == {Change.SYSTEM_MODEL}
    imp = impact(changes)
    assert {"agent", "reliability", "regression"} <= imp.suites
    assert imp.rerecord_outputs and imp.rerecord_verdicts
    assert imp.revalidate_judges


def test_judge_only_change_keeps_outputs() -> None:
    after = BASE.model_copy(update={"judge_model": "j-2026-09"})
    imp = impact(detect(BASE, after))
    assert not imp.rerecord_outputs
    assert imp.rerecord_verdicts and imp.revalidate_judges
    assert imp.suites == {"judge-validation", "regression"}


def test_no_change_runs_nothing() -> None:
    imp = impact(detect(BASE, BASE))
    assert imp.suites == set() and not imp.rerecord_outputs


def test_regression_case() -> None:
    c = regression_case(
        "reg-017", "Parcel lost. 30 days from when?",
        ["OBJ-CLAIM-01"], "FM-DEADLINE-START",
        Reference(kind=ReferenceKind.CRITERIA,
                  values=["Counts from expected delivery date"]),
        Source.PRODUCTION)
    assert "fm:FM-DEADLINE-START" in c.tags and c.reviewed
    assert c.split is Split.DEV


def test_sandbox_change_rerecords_agent_runs() -> None:
    after = BASE.model_copy(update={"sandbox": "s2"})
    imp = impact(detect(BASE, after))
    assert imp.rerecord_outputs
    assert imp.suites == {"agent", "regression"}
