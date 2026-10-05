from pathlib import Path

import pytest

from aiqe.datasets import (EvalCase, Reference, ReferenceKind,
                           Source, Split, load_jsonl)
from aiqe.evaluators.code import (ClaimDeadline,
                                  CommitmentPreFilter,
                                  ContactAllowList, same_deadline)
from aiqe.runner import run


def case(tags: list[str]) -> EvalCase:
    return EvalCase(case_id="c", input="q",
                    objective_ids=["OBJ-CLAIM-01"],
                    reference=Reference(kind=ReferenceKind.EXACT,
                                        values=["x"]),
                    source=Source.HANDWRITTEN, tags=tags)


@pytest.mark.parametrize("out,ok", [
    ("Report damage within 14 days of delivery.", True),
    ("You have fourteen days from delivery.", True),
    ("Report damage within 7 days.", False),
    ("Please contact us soon.", False),
])
def test_damage_deadline(out: str, ok: bool) -> None:
    s = ClaimDeadline().evaluate(case(["claim", "damage"]), out)
    assert s.passed is ok


def test_loss_start_event() -> None:
    ev, c = ClaimDeadline(), case(["claim", "loss"])
    good = "Claim within 30 days of the expected delivery date."
    bad = "Claim within 30 days from your order date."
    assert ev.evaluate(c, good).passed is True
    s = ev.evaluate(c, bad)
    assert s.passed is False and "order" in s.detail


def test_not_a_claim_case_is_deferred() -> None:
    s = ClaimDeadline().evaluate(case(["credit"]), "anything")
    assert s.passed is None


def test_contact_allow_list() -> None:
    ev = ContactAllowList(
        emails={"claims@tidewater-logistics.example"},
        phones={"020 7946 0321"},
        urls={"https://tidewater-logistics.example/claims"})
    c = case(["claim"])
    ok = ("Email claims@tidewater-logistics.example or call "
          "+44 20 7946 0321.")
    assert ev.evaluate(c, ok).passed is True
    bad = "Call our claims line on 020 7946 0999."
    assert ev.evaluate(c, bad).passed is False
    url = "See https://tidewater-logistics.example/claims."
    assert ev.evaluate(c, url).passed is True


def test_commitment_pre_filter_fails_or_defers() -> None:
    ev, c = CommitmentPreFilter(), case(["credit"])
    assert ev.evaluate(c, "You'll receive a credit.").passed \
        is False
    assert ev.evaluate(c, "A credit may apply.").passed is None


def test_metamorphic_same_deadline() -> None:
    assert same_deadline(["14 days from delivery",
                          "fourteen days after delivery"])
    assert not same_deadline(["14 days", "7 days"])


def test_runner_uses_split_and_records_hash() -> None:
    ds = load_jsonl(
        Path("datasets/compass/credit-claims-v1.jsonl"),
        "credit-claims", "1.0")
    res = run(ds, Split.HOLDOUT, "stub",
              lambda q: "Report damage within 14 days of delivery.",
              [ClaimDeadline()])
    assert res.dataset_hash == ds.content_hash()
    assert set(res.outputs) == {"cc-001", "cc-002", "cc-003"}
    assert res.counts("claim-deadline") == {
        "pass": 1, "fail": 0, "deferred": 2}
