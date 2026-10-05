"""aiqe.evaluators: the shared evaluator interface."""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from aiqe.datasets import EvalCase


class Score(BaseModel):
    """One evaluator's judgement of one output.

    passed is None when the evaluator cannot reach a verdict
    and the case must be deferred to another evaluator.
    """

    case_id: str
    evaluator: str
    passed: bool | None
    detail: str = ""


class Evaluator(Protocol):
    name: str

    def evaluate(self, case: EvalCase, output: str) -> Score:
        ...
