"""aiqe/experiments.py: baselines, candidates, and decisions."""
from __future__ import annotations

import json
from datetime import date
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from aiqe.datasets import Split
from aiqe.runner import RunResult
from aiqe.stats import Paired, paired_compare


class BaselineStore:
    """Named, immutable runs that candidates are compared with."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def promote(self, run: RunResult, label: str) -> Path:
        path = self.root / f"{label}.json"
        if path.exists():
            raise FileExistsError(f"baseline {label} exists")
        path.write_text(run.model_dump_json(indent=1))
        return path

    def load(self, label: str) -> RunResult:
        path = self.root / f"{label}.json"
        return RunResult.model_validate_json(path.read_text())


def verdicts(run: RunResult, evaluator: str) -> dict[str, bool]:
    return {s.case_id: s.passed for s in run.scores
            if s.evaluator == evaluator and s.passed is not None}


def compare_runs(baseline: RunResult, candidate: RunResult,
                 evaluator: str) -> Paired:
    """Paired comparison; refuses mismatched evidence."""
    if baseline.dataset_hash != candidate.dataset_hash:
        raise ValueError("runs used different dataset content")
    if baseline.split is not candidate.split:
        raise ValueError("runs used different splits")
    return paired_compare(verdicts(baseline, evaluator),
                          verdicts(candidate, evaluator))


class Outcome(StrEnum):
    ADOPT = "adopt"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"


class Plan(BaseModel):
    """Written before the candidate is evaluated on holdout."""

    hypothesis: str
    primary: str                  # evaluator expected to improve
    guards: list[str]             # evaluators that must not regress
    alpha: float = 0.05


class ExperimentRecord(BaseModel):
    experiment_id: str
    plan: Plan
    baseline: str                 # label in the BaselineStore
    candidate: str                # system name of the candidate
    dataset_hash: str
    split: Split
    results: dict[str, Paired]
    outcome: Outcome
    regressed_cases: dict[str, list[str]]
    recorded_on: date = Field(default_factory=date.today)


def regressions(baseline: RunResult, candidate: RunResult,
                evaluator: str) -> list[str]:
    b, c = verdicts(baseline, evaluator), verdicts(candidate,
                                                   evaluator)
    return sorted(k for k in b.keys() & c.keys()
                  if b[k] and not c[k])


def decide(plan: Plan, results: dict[str, Paired]) -> Outcome:
    for g in plan.guards:
        r = results[g]
        if r.difference.estimate < 0 and r.p_value < plan.alpha:
            return Outcome.REJECT
    p = results[plan.primary]
    if p.difference.estimate > 0 and p.p_value < plan.alpha:
        return Outcome.ADOPT
    return Outcome.INCONCLUSIVE


def run_experiment(experiment_id: str, plan: Plan,
                   store: BaselineStore, baseline_label: str,
                   candidate: RunResult) -> ExperimentRecord:
    base = store.load(baseline_label)
    names = [plan.primary, *plan.guards]
    results = {n: compare_runs(base, candidate, n)
               for n in names}
    return ExperimentRecord(
        experiment_id=experiment_id, plan=plan,
        baseline=baseline_label, candidate=candidate.system,
        dataset_hash=candidate.dataset_hash,
        split=candidate.split, results=results,
        outcome=decide(plan, results),
        regressed_cases={n: regressions(base, candidate, n)
                         for n in names})


class HoldoutLedger:
    """Counts holdout evaluations per dataset version."""

    def __init__(self, path: Path, limit: int) -> None:
        self.path = path
        self.limit = limit

    def use(self, dataset_hash: str, experiment_id: str) -> int:
        data: dict[str, list[str]] = (
            json.loads(self.path.read_text())
            if self.path.exists() else {})
        uses = data.setdefault(dataset_hash, [])
        uses.append(experiment_id)
        self.path.write_text(json.dumps(data, indent=1))
        if len(uses) > self.limit:
            raise RuntimeError(
                f"holdout {dataset_hash} used {len(uses)} times; "
                "refresh it before further decisions")
        return len(uses)
