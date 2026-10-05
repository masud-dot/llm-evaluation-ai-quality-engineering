"""aiqe/trials.py: repeated trials under replay."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from aiqe.datasets import Dataset, Split
from aiqe.evaluators import Evaluator
from aiqe.providers import ReplayProvider
from aiqe.stats import Trials


def trial_providers(root: Path, n: int) -> list[ReplayProvider]:
    """One replay directory per trial: trial-0, trial-1, ...

    Each trial is recorded separately. Replaying one recording
    n times would show zero variance and prove nothing.
    """
    dirs = [root / f"trial-{i}" for i in range(n)]
    missing = [d.name for d in dirs if not d.is_dir()]
    if missing:
        raise FileNotFoundError(f"no recordings for {missing}")
    return [ReplayProvider(d) for d in dirs]


def run_trials(dataset: Dataset, split: Split,
               systems: Sequence[Callable[[str], str]],
               evaluator: Evaluator) -> tuple[Trials, int]:
    """Verdicts per case across trials, and deferrals dropped."""
    trials: Trials = {}
    deferred = 0
    for case in (c for c in dataset.cases if c.split is split):
        verdicts = []
        for system in systems:
            s = evaluator.evaluate(case, system(case.input))
            if s.passed is None:
                deferred += 1
            else:
                verdicts.append(s.passed)
        if verdicts:
            trials[case.case_id] = verdicts
    return trials, deferred


def flaky(trials: Trials) -> list[str]:
    """Cases whose verdict differs between trials."""
    return sorted(c for c, v in trials.items()
                  if len(set(v)) > 1)
