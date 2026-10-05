"""aiqe/triage.py: from production signal to regression case."""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from aiqe.changes import regression_case
from aiqe.datasets import (Dataset, EvalCase, Reference, Source,
                           pii_suspects)


class Status(StrEnum):
    NEW = "new"
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"
    PROMOTED = "promoted"


ALLOWED = {Status.NEW: {Status.CONFIRMED, Status.DISMISSED},
           Status.CONFIRMED: {Status.PROMOTED},
           Status.DISMISSED: set(), Status.PROMOTED: set()}


class TriageItem(BaseModel):
    item_id: str
    conversation_id: str
    signal: str               # what raised it
    status: Status = Status.NEW
    failure_mode: str | None = None
    objective_ids: list[str] = []
    reviewer: str | None = None
    note: str = ""

    def move(self, to: Status, reviewer: str,
             note: str) -> TriageItem:
        if to not in ALLOWED[self.status]:
            raise ValueError(f"{self.status} -> {to} not allowed")
        if to is Status.CONFIRMED and not (
                self.failure_mode and self.objective_ids):
            raise ValueError("confirm needs mode and objectives")
        return self.model_copy(update={
            "status": to, "reviewer": reviewer, "note": note})


def promote(item: TriageItem, deidentified_input: str,
            reference: Reference) -> tuple[TriageItem, EvalCase]:
    """Turn a confirmed failure into a regression case."""
    if item.status is not Status.CONFIRMED or not item.failure_mode:
        raise ValueError("only confirmed failures are promoted")
    case = regression_case(
        f"reg-{item.item_id}", deidentified_input,
        item.objective_ids, item.failure_mode, reference,
        Source.PRODUCTION)
    probe = Dataset(name="probe", version="0", cases=[case])
    if pii_suspects(probe):
        raise ValueError("input still looks like personal data")
    done = item.move(Status.PROMOTED, item.reviewer or "",
                     f"promoted as {case.case_id}")
    return done, case
