"""aiqe/evaluators/tasks.py: extraction and classification."""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from aiqe.datasets import EvalCase
from aiqe.evaluators import Score


class ClaimIntake(BaseModel):
    """The structured record Compass fills for a new claim."""

    shipment_id: str = Field(pattern=r"^TW-\d{4,8}$")
    claim_type: Literal["damage", "loss"]
    delivery_date: date | None = None
    items: list[str] = Field(default_factory=list)


class IntakeSchema:
    """Structural check: output parses as a ClaimIntake."""

    name = "intake-schema"

    def evaluate(self, case: EvalCase, output: str) -> Score:
        try:
            ClaimIntake.model_validate(json.loads(output))
        except (json.JSONDecodeError, ValidationError) as e:
            first = str(e).splitlines()[0]
            return Score(case_id=case.case_id,
                         evaluator=self.name, passed=False,
                         detail=first)
        return Score(case_id=case.case_id, evaluator=self.name,
                     passed=True)


def field_matches(pred: ClaimIntake, ref: ClaimIntake,
                  fields: Iterable[str]) -> dict[str, bool]:
    """Per-field correctness; items compared as sets."""
    out = {}
    for f in fields:
        a, b = getattr(pred, f), getattr(ref, f)
        if f == "items":
            a = {x.strip().lower() for x in a}
            b = {x.strip().lower() for x in b}
        out[f] = a == b
    return out


class PRF(BaseModel):
    precision: float | None
    recall: float | None
    f1: float | None


def prf(pred: set[str], ref: set[str]) -> PRF:
    """Precision, recall, F1 for a set of extracted items."""
    tp = len(pred & ref)
    p = tp / len(pred) if pred else None
    r = tp / len(ref) if ref else None
    f1 = (2 * p * r / (p + r)) if p and r else (
        0.0 if p is not None and r is not None else None)
    return PRF(precision=p, recall=r, f1=f1)


Intent = Literal["tracking", "claim", "credit", "billing",
                 "other"]


def per_class_recall(pairs: Iterable[tuple[Intent, Intent]]
                     ) -> dict[str, tuple[int, int]]:
    """(correct, total) per true class from (true, predicted)."""
    total: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for truth, pred in pairs:
        total[truth] += 1
        correct[truth] += truth == pred
    return {k: (correct[k], total[k]) for k in sorted(total)}
