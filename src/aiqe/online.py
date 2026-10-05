"""aiqe/online.py: evaluating sampled production traffic."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from aiqe.datasets import EvalCase, Source
from aiqe.evaluators import Evaluator, Score


class OnlineScore(BaseModel):
    """A score from production, never mixed with offline."""

    conversation_id: str
    evaluated_at: datetime
    setting: Literal["online"] = "online"
    score: Score


def evaluate_sampled(conversation_id: str, customer_text: str,
                     output: str, objective_ids: list[str],
                     evaluators: Sequence[Evaluator],
                     now: datetime) -> list[OnlineScore]:
    """Apply reference-free evaluators to one sampled answer.

    Production has no references, so only evaluators that need
    none (invariants, pre-filters, validated judges) belong here.
    """
    case = EvalCase(case_id=conversation_id, input=customer_text,
                    objective_ids=objective_ids,
                    source=Source.PRODUCTION)
    return [OnlineScore(conversation_id=conversation_id,
                        evaluated_at=now,
                        score=e.evaluate(case, output))
            for e in evaluators]
