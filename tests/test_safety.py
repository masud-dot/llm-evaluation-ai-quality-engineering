import pytest
from pydantic import ValidationError

from aiqe.datasets import Source
from aiqe.safety import (Category, SafetyCase, canary_check,
                         refusal_matrix)


def case(cid: str, cat: Category, refuse: bool,
         canaries: list[str] | None = None) -> SafetyCase:
    src = (Source.HANDWRITTEN if cat is Category.BENIGN
           else Source.ADVERSARIAL)
    return SafetyCase(case_id=cid, input="q",
                      objective_ids=["OBJ-SAFE-01"], source=src,
                      category=cat, should_refuse=refuse,
                      canaries=canaries or [])


def test_case_consistency() -> None:
    with pytest.raises(ValidationError):
        case("a", Category.BENIGN, True)
    with pytest.raises(ValidationError):
        case("b", Category.POLICY, False)


def test_canary() -> None:
    c = case("leak-1", Category.LEAKAGE, True,
             ["CANARY-7Q2F", "ADDR-7"])
    assert canary_check(c, "Your parcel is on its way.").passed
    s = canary_check(c, "TW-2210 goes to ADDR-7.")
    assert s.passed is False and "ADDR-7" in s.detail
    plain = case("p", Category.POLICY, True)
    assert canary_check(plain, "x").passed is None


def test_refusal_matrix() -> None:
    cases = [case("h1", Category.POLICY, True),
             case("h2", Category.INJECTION, True),
             case("b1", Category.BENIGN, False),
             case("b2", Category.BENIGN, False),
             case("b3", Category.BENIGN, False)]
    m = refusal_matrix(cases, {"h1": True, "h2": False,
                               "b1": True, "b2": False,
                               "b3": None})
    assert m.unsafe_rate == 0.5
    assert m.over_refusal_rate == 0.5
    assert m.undecided == 1
