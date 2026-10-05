from collections import Counter

import pytest
from pydantic import ValidationError

from aiqe.review import (UNCODED, Annotation, FailureMode,
                         Severity, Taxonomy, Trace, Verdict,
                         prioritize, review_session, tally)

TAX = Taxonomy(version="0.1", modes=[
    FailureMode(id="FM-CREDIT-PROMISE", name="Credit promise",
                definition="d", include=["e"],
                severity=Severity.CRITICAL,
                objective_ids=["OBJ-CREDIT-01"]),
    FailureMode(id="FM-NO-NEXT-STEP", name="No next step",
                definition="d", include=["e"],
                severity=Severity.STANDARD,
                objective_ids=["OBJ-HELP-01"]),
    FailureMode(id="FM-INVENTED-CONTACT", name="Invented contact",
                definition="d", include=["e"],
                severity=Severity.HIGH),
])


def ann(i: str, v: Verdict, mode: str | None = None) -> Annotation:
    return Annotation(trace_id=i, case_id=i, reviewer="r",
                      verdict=v, note="n", failure_mode=mode)


def test_fail_requires_note() -> None:
    with pytest.raises(ValidationError):
        Annotation(trace_id="t", case_id="c", reviewer="r",
                   verdict=Verdict.FAIL)


def test_review_session_scripted() -> None:
    answers = iter(["fail", "promised a credit", "pass"])
    traces = [Trace(trace_id=f"t{i}", case_id=f"c{i}",
                    input="q", output="a", model="m",
                    prompt_version="p1") for i in (1, 2)]
    out = review_session(traces, "r", lambda _: next(answers))
    assert [a.verdict for a in out] == [Verdict.FAIL,
                                        Verdict.PASS]
    assert out[0].note == "promised a credit"


def test_tally_prioritize_and_gaps() -> None:
    anns = ([ann(f"a{i}", Verdict.FAIL, "FM-NO-NEXT-STEP")
             for i in range(6)]
            + [ann("b", Verdict.FAIL, "FM-CREDIT-PROMISE"),
               ann("c", Verdict.FAIL),
               ann("d", Verdict.PASS)])
    counts = tally(anns, TAX)
    assert counts == Counter({"FM-NO-NEXT-STEP": 6,
                              "FM-CREDIT-PROMISE": 1,
                              UNCODED: 1})
    order = [r[0] for r in prioritize(counts, TAX)]
    assert order == ["FM-CREDIT-PROMISE", "FM-NO-NEXT-STEP"]
    assert TAX.gaps() == ["FM-INVENTED-CONTACT"]


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError):
        tally([ann("x", Verdict.FAIL, "FM-NOPE")], TAX)
