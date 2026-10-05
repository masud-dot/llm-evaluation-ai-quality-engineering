"""aiqe/policy.py: release gate policies over evidence."""
from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from aiqe.stats import Paired


class Severity(StrEnum):
    BLOCK = "block"       # hard constraint
    REVIEW = "review"     # trade-off needing a recorded decision


class Limit(BaseModel):
    kind: Literal["max", "min"]
    metric: str           # a bound, e.g. an interval's upper end
    value: float
    severity: Severity
    objective: str


class NoRegression(BaseModel):
    evaluator: str
    alpha: float = 0.05
    severity: Severity
    objective: str


class GatePolicy(BaseModel):
    name: str
    version: str
    limits: list[Limit]
    no_regression: list[NoRegression]


class Evidence(BaseModel):
    values: dict[str, float]
    paired: dict[str, Paired]


class Verdict(StrEnum):
    PASS = "pass"
    REVIEW = "review"
    FAIL = "fail"


class Finding(BaseModel):
    objective: str
    severity: Severity
    message: str


class GateResult(BaseModel):
    policy: str
    verdict: Verdict
    findings: list[Finding]


def load_policy(path: Path) -> GatePolicy:
    with path.open("rb") as fh:
        return GatePolicy.model_validate(tomllib.load(fh))


def apply(policy: GatePolicy, ev: Evidence) -> GateResult:
    """Missing evidence is a finding, never a pass."""
    found: list[Finding] = []
    for lim in policy.limits:
        v = ev.values.get(lim.metric)
        if v is None:
            msg = f"no evidence for {lim.metric}"
        elif (v > lim.value) if lim.kind == "max" \
                else (v < lim.value):
            msg = f"{lim.metric}={v:.4g} breaks {lim.value}"
        else:
            continue
        found.append(Finding(objective=lim.objective,
                             severity=lim.severity, message=msg))
    for nr in policy.no_regression:
        p = ev.paired.get(nr.evaluator)
        if p is None:
            msg = f"no paired evidence for {nr.evaluator}"
        elif p.difference.estimate < 0 and p.p_value < nr.alpha:
            msg = (f"{nr.evaluator} regressed "
                   f"({p.difference.estimate:+.3f}, "
                   f"p={p.p_value:.3g})")
        else:
            continue
        found.append(Finding(objective=nr.objective,
                             severity=nr.severity, message=msg))
    if any(f.severity is Severity.BLOCK for f in found):
        verdict = Verdict.FAIL
    elif found:
        verdict = Verdict.REVIEW
    else:
        verdict = Verdict.PASS
    return GateResult(policy=f"{policy.name} v{policy.version}",
                      verdict=verdict, findings=found)
