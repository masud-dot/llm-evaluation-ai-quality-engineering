"""aiqe/metrics.py: from scores to metrics, overall and sliced."""
from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel

from aiqe.datasets import Dataset
from aiqe.evaluators import Score
from aiqe.runner import RunResult


class Rate(BaseModel):
    """A pass rate that never hides its denominator."""

    evaluator: str
    passed: int
    failed: int
    deferred: int

    @property
    def decided(self) -> int:
        return self.passed + self.failed

    @property
    def value(self) -> float | None:
        """Pass rate over decided cases; None if none decided."""
        return self.passed / self.decided if self.decided else None

    @property
    def coverage(self) -> float | None:
        """Share of cases the evaluator could decide."""
        total = self.decided + self.deferred
        return self.decided / total if total else None


def pass_rate(scores: Iterable[Score], evaluator: str) -> Rate:
    p = f = d = 0
    for s in scores:
        if s.evaluator != evaluator:
            continue
        if s.passed is None:
            d += 1
        elif s.passed:
            p += 1
        else:
            f += 1
    return Rate(evaluator=evaluator, passed=p, failed=f,
                deferred=d)


def sliced(run: RunResult, dataset: Dataset, evaluator: str,
           tags: Iterable[str]) -> dict[str, Rate]:
    """One Rate per tag, over the cases carrying that tag."""
    tags_by_case = {c.case_id: set(c.tags)
                    for c in dataset.cases}
    out: dict[str, Rate] = {}
    for tag in tags:
        in_slice = [s for s in run.scores
                    if tag in tags_by_case.get(s.case_id, set())]
        out[tag] = pass_rate(in_slice, evaluator)
    return out
