"""Run the routed Compass suites on replay and write evidence.

Baseline evidence is carried only when the change leaves system
outputs untouched; once outputs change, stale evidence is not
evidence and every value must be measured again. A missing
recording or an unconfigured routed suite raises, so the gate
never passes on it.
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from aiqe.changes import Impact
from aiqe.datasets import Split, load_jsonl
from aiqe.evaluators import Evaluator
from aiqe.evaluators.code import (ClaimDeadline,
                                  CommitmentPreFilter,
                                  ContactAllowList)
from aiqe.evaluators.judge import Cascade, JudgeConfig, SingleJudge
from aiqe.evidence import EvidenceBuilder
from aiqe.experiments import compare_runs
from aiqe.metrics import pass_rate
from aiqe.policy import Evidence
from aiqe.providers import ReplayProvider
from aiqe.rubrics import Rubric
from aiqe.runner import RunResult, run
from systems.compass.single_turn import SingleTurnCompass


class SystemSpec(BaseModel):
    model: str
    prompt: str
    replay: str
    judge_replay: str
    judge: str
    rubric: str
    emails: list[str]
    phones: list[str]
    urls: list[str]


class SuiteSpec(BaseModel):
    name: str
    kind: Literal["critical", "regression"]
    dataset: str
    dataset_name: str
    version: str
    split: Split
    tiers: list[str]


class SuitesConfig(BaseModel):
    system: SystemSpec
    suites: list[SuiteSpec]


def load_config(path: Path) -> SuitesConfig:
    with path.open("rb") as fh:
        return SuitesConfig.model_validate(tomllib.load(fh))


def evaluators(root: Path, s: SystemSpec) -> list[Evaluator]:
    with (root / s.judge).open("rb") as fh:
        cfg = JudgeConfig.model_validate(tomllib.load(fh))
    with (root / s.rubric).open("rb") as fh:
        rubric = Rubric.model_validate(tomllib.load(fh))
    cfg = cfg.model_copy(update={
        "template_path": str(root / cfg.template_path)})
    judge = SingleJudge(ReplayProvider(root / s.judge_replay),
                        cfg, rubric)
    return [ClaimDeadline(),
            ContactAllowList(set(s.emails), set(s.phones),
                             set(s.urls)),
            Cascade("credit-cascade", CommitmentPreFilter(),
                    judge)]


def fails(r: RunResult, evaluator: str) -> int:
    return sum(1 for x in r.scores
               if x.evaluator == evaluator and x.passed is False)


def run_suites(root: Path, config: Path, impact: Impact,
               tier: str, baseline: Path,
               evidence_path: Path) -> Evidence:
    cfg = load_config(config)
    carried = Evidence.model_validate_json(
        (baseline / "evidence.json").read_text())
    ev = EvidenceBuilder()
    if not impact.rerecord_outputs:
        for k, v in carried.values.items():
            ev.value(k, v)
        for k, p in carried.paired.items():
            ev.comparison(k, p)
    system = SingleTurnCompass.from_files(
        ReplayProvider(root / cfg.system.replay),
        cfg.system.model, root / cfg.system.prompt)
    evs = evaluators(root, cfg.system)
    configured = {x.name for x in cfg.suites if tier in x.tiers}
    missing = sorted(impact.suites - configured)
    if missing:
        raise ValueError(f"routed suites not configured for "
                         f"tier {tier}: {missing}")
    out = evidence_path.parent
    out.mkdir(parents=True, exist_ok=True)
    for suite in cfg.suites:
        if suite.name not in impact.suites or \
                tier not in suite.tiers:
            continue
        ds = load_jsonl(root / suite.dataset, suite.dataset_name,
                        suite.version)
        res = run(ds, suite.split, "candidate", system, evs)
        (out / f"run-{suite.name}.json").write_text(
            res.model_dump_json(indent=1))
        if suite.kind == "regression":
            ev.count("regression.failures",
                     sum(fails(res, e.name) for e in evs))
            continue
        ev.count("OBJ-CLAIM-01.failures",
                 fails(res, "claim-deadline"))
        ev.count("OBJ-CONTACT-01.failures",
                 fails(res, "contact-allow-list"))
        ev.failure_rate("OBJ-CREDIT-01",
                        pass_rate(res.scores, "credit-cascade"))
        base_file = baseline / f"run-{suite.name}.json"
        if not base_file.exists():
            continue    # first baseline: no comparison yet
        base = RunResult.model_validate_json(base_file.read_text())
        for name in ["claim-deadline", "credit-cascade"]:
            ev.comparison(name, compare_runs(base, res, name))
    evidence = ev.build()
    evidence_path.write_text(evidence.model_dump_json(indent=1))
    return evidence


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impact", type=Path, required=True)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=Path("suites/compass-suites.toml"))
    ap.add_argument("--baseline", type=Path,
                    default=Path("baselines/current"))
    a = ap.parse_args(argv)
    imp = Impact.model_validate_json(a.impact.read_text())
    run_suites(Path.cwd(), a.config, imp, a.tier, a.baseline,
               a.evidence)
    return 0


if __name__ == "__main__":
    sys.exit(main())
