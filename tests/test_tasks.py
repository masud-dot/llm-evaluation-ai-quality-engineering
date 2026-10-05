from collections.abc import Callable

import pytest

from aiqe.conversation import (ConversationCase, Expectation,
                               Message, check, play, render)
from aiqe.datasets import EvalCase, Source
from aiqe.evaluators.tasks import (ClaimIntake, IntakeSchema,
                                   Intent, field_matches,
                                   per_class_recall, prf)

CASE = EvalCase(case_id="x-1", input="q",
                objective_ids=["OBJ-CLAIM-01"],
                source=Source.HANDWRITTEN)


def test_intake_schema() -> None:
    ev = IntakeSchema()
    ok = ('{"shipment_id": "TW-1042", "claim_type": "damage",'
          ' "delivery_date": "2026-09-01", "items": ["lamp"]}')
    assert ev.evaluate(CASE, ok).passed is True
    bad = '{"shipment_id": "1042", "claim_type": "damage"}'
    s = ev.evaluate(CASE, bad)
    assert s.passed is False and s.detail
    assert ev.evaluate(CASE, "not json").passed is False


def test_field_matches_and_prf() -> None:
    ref = ClaimIntake(shipment_id="TW-1042", claim_type="damage",
                      items=["Lamp", "rug"])
    pred = ClaimIntake(shipment_id="TW-1042", claim_type="loss",
                       items=["lamp ", "RUG"])
    m = field_matches(pred, ref,
                      ["shipment_id", "claim_type", "items"])
    assert m == {"shipment_id": True, "claim_type": False,
                 "items": True}
    r = prf({"lamp", "rug", "vase"}, {"lamp", "rug"})
    assert r.precision == pytest.approx(2 / 3)
    assert r.recall == 1.0
    assert r.f1 == pytest.approx(0.8)
    assert prf(set(), set()).f1 is None
    assert prf({"a"}, {"b"}).f1 == 0.0


def test_per_class_recall() -> None:
    pairs: list[tuple[Intent, Intent]] = [
        ("claim", "claim"), ("claim", "credit"),
        ("credit", "credit"), ("other", "claim")]
    assert per_class_recall(pairs) == {
        "claim": (1, 2), "credit": (1, 1), "other": (0, 1)}


def test_conversation_state_is_tracked() -> None:
    case = ConversationCase(
        case_id="conv-1", objective_ids=["OBJ-CLAIM-01"],
        user_turns=["My parcel TW-1042 arrived damaged.",
                    "What happens next?"],
        expectations=[Expectation(
            turn=1, must_include=["TW-1042", "14 days"],
            must_not_include=["30 days"])])
    good = ["Sorry to hear that.",
            "For TW-1042, claim within 14 days of delivery."]
    forgetful = ["Sorry to hear that.",
                 "Which shipment? Claims take 30 days."]
    seen: list[int] = []

    def sys_from(
            replies: list[str]) -> Callable[[list[Message]], str]:
        def system(history: list[Message]) -> str:
            seen.append(len(history))
            return replies[len(history) // 2]
        return system

    assert check(case, play(case, sys_from(good)))[0].passed
    bad = check(case, play(case, sys_from(forgetful)))[0]
    assert not bad.passed
    assert bad.missing == ["TW-1042", "14 days"]
    assert bad.forbidden == ["30 days"]
    assert seen[:2] == [1, 3]
    assert render([Message(role="user", content="hi")]) == \
        "user: hi"
