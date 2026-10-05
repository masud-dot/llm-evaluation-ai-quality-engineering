from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from aiqe.governance import (CRITERIA, Level, ReleaseRecord,
                             Signature, assess, evidence_digest,
                             required_signers)
from aiqe.strategy import load_plan

PLAN = load_plan(Path("plans/compass-plan.toml"))
REQ = required_signers(PLAN)


def record(sigs: list[Signature],
           outcome: str = "pass") -> ReleaseRecord:
    return ReleaseRecord(
        release_id="compass-2026.10", system="compass",
        plan_version=PLAN.version,
        policy="compass-release v1.2", gate_outcome=outcome,
        evidence_digest="0" * 16, overrides=[],
        signatures=sigs, required=REQ,
        delegations={"Head of Credit Control": "Priya Shah"},
        released_on=date(2026, 10, 1))


def test_required_signers_are_critical_owners() -> None:
    assert "Head of Credit Control" in REQ
    assert "Data Protection Officer" in REQ
    assert "Platform Engineering Lead" not in REQ


def test_release_needs_every_owner_or_recorded_deputy() -> None:
    sigs = [Signature(role=r, person="x") for r in REQ]
    assert record(sigs).release_id
    with pytest.raises(ValidationError):
        record(sigs[1:])
    deputy = [Signature(role="deputy", person="Priya Shah",
                        on_behalf_of="Head of Credit Control")]
    others = [Signature(role=r, person="x") for r in REQ
              if r != "Head of Credit Control"]
    assert record(deputy + others)
    wrong = [Signature(role="deputy", person="Sam Ode",
                       on_behalf_of="Head of Credit Control")]
    with pytest.raises(ValidationError):
        record(wrong + others)
    with pytest.raises(ValidationError):
        record(sigs, outcome="review")


def test_evidence_digest_is_order_independent(
        tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text("1")
    b.write_text("2")
    before = evidence_digest([a, b])
    assert before == evidence_digest([b, a])
    b.write_text("3")
    assert evidence_digest([a, b]) != before


def test_maturity_is_cumulative() -> None:
    assert assess(set()) is Level.AD_HOC
    two_and_four = CRITERIA[Level.DEFINED] | CRITERIA[Level.GATED]
    assert assess(two_and_four) is Level.DEFINED
    everything = set().union(*CRITERIA.values())
    assert assess(everything) is Level.CONTINUOUS
