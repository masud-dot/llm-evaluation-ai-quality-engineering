from pathlib import Path

import pytest

from aiqe.datasets import Split
from aiqe.evaluators import Score
from aiqe.experiments import (BaselineStore, HoldoutLedger,
                              Outcome, Plan, compare_runs,
                              run_experiment)
from aiqe.runner import RunResult


def mk(system: str, claim: dict[str, bool],
       credit: dict[str, bool], h: str = "h1") -> RunResult:
    scores = [Score(case_id=k, evaluator="claim", passed=v)
              for k, v in claim.items()]
    scores += [Score(case_id=k, evaluator="credit", passed=v)
               for k, v in credit.items()]
    return RunResult(dataset="d", dataset_version="1",
                     dataset_hash=h, split=Split.HOLDOUT,
                     system=system, outputs={}, scores=scores)


IDS = [f"c{i}" for i in range(80)]
BASE_CLAIM = {k: i >= 20 for i, k in enumerate(IDS)}
ALL_PASS = {k: True for k in IDS}
PLAN = Plan(hypothesis="new prompt fixes loss deadlines",
            primary="claim", guards=["credit"])


def store(tmp: Path) -> BaselineStore:
    s = BaselineStore(tmp / "baselines")
    s.promote(mk("prompt-v3", BASE_CLAIM, ALL_PASS), "v3")
    return s


def test_promote_is_immutable(tmp_path: Path) -> None:
    s = store(tmp_path)
    with pytest.raises(FileExistsError):
        s.promote(mk("x", {}, {}), "v3")


def test_adopt_when_primary_improves(tmp_path: Path) -> None:
    cand = mk("prompt-v4", ALL_PASS, ALL_PASS)
    rec = run_experiment("exp-1", PLAN, store(tmp_path), "v3",
                         cand)
    assert rec.outcome is Outcome.ADOPT
    assert rec.regressed_cases == {"claim": [], "credit": []}


def test_reject_when_guard_regresses(tmp_path: Path) -> None:
    credit = {k: i >= 15 for i, k in enumerate(IDS)}
    cand = mk("prompt-v4", ALL_PASS, credit)
    rec = run_experiment("exp-2", PLAN, store(tmp_path), "v3",
                         cand)
    assert rec.outcome is Outcome.REJECT
    assert len(rec.regressed_cases["credit"]) == 15


def test_inconclusive_and_mismatch(tmp_path: Path) -> None:
    claim = dict(BASE_CLAIM)
    claim["c0"] = True
    cand = mk("prompt-v4", claim, ALL_PASS)
    rec = run_experiment("exp-3", PLAN, store(tmp_path), "v3",
                         cand)
    assert rec.outcome is Outcome.INCONCLUSIVE
    other = mk("prompt-v4", ALL_PASS, ALL_PASS, h="h2")
    with pytest.raises(ValueError):
        compare_runs(store(tmp_path / "x").load("v3"), other,
                     "claim")


def test_holdout_ledger(tmp_path: Path) -> None:
    ledger = HoldoutLedger(tmp_path / "ledger.json", limit=2)
    assert ledger.use("h1", "exp-1") == 1
    assert ledger.use("h1", "exp-2") == 2
    with pytest.raises(RuntimeError):
        ledger.use("h1", "exp-3")
    assert ledger.use("h2", "exp-4") == 1
