"""aiqe/runner.py: run a system and evaluators over a dataset."""
from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel

from aiqe.datasets import Dataset, Split
from aiqe.evaluators import Evaluator, Score


class RunResult(BaseModel):
    dataset: str
    dataset_version: str
    dataset_hash: str
    split: Split
    system: str
    outputs: dict[str, str]
    scores: list[Score]

    def counts(self, evaluator: str) -> dict[str, int]:
        c = {"pass": 0, "fail": 0, "deferred": 0}
        for s in self.scores:
            if s.evaluator != evaluator:
                continue
            key = ("deferred" if s.passed is None
                   else "pass" if s.passed else "fail")
            c[key] += 1
        return c


def run(dataset: Dataset, split: Split, system_name: str,
        system: Callable[[str], str],
        evaluators: Sequence[Evaluator]) -> RunResult:
    cases = [c for c in dataset.cases if c.split is split]
    outputs: dict[str, str] = {}
    scores: list[Score] = []
    for case in cases:
        out = system(case.input)
        outputs[case.case_id] = out
        scores.extend(e.evaluate(case, out) for e in evaluators)
    return RunResult(
        dataset=dataset.name, dataset_version=dataset.version,
        dataset_hash=dataset.content_hash(), split=split,
        system=system_name, outputs=outputs, scores=scores)
