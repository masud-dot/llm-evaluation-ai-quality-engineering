"""aiqe/governance.py: sign-off, evidence, and maturity."""
from __future__ import annotations

import hashlib
from datetime import date
from enum import IntEnum
from pathlib import Path

from pydantic import BaseModel, model_validator

from aiqe.strategy import EvaluationPlan, RiskTier


def evidence_digest(paths: list[Path]) -> str:
    """One hash over the files a release decision relied on."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def required_signers(plan: EvaluationPlan) -> set[str]:
    """Every owner of a critical objective signs the release."""
    return {o.owner for o in plan.by_risk(RiskTier.CRITICAL)}


class Signature(BaseModel):
    role: str            # as named in the plan
    person: str
    on_behalf_of: str | None = None   # recorded delegation


class ReleaseRecord(BaseModel):
    release_id: str
    system: str
    plan_version: str
    policy: str
    gate_outcome: str
    evidence_digest: str
    overrides: list[str]
    signatures: list[Signature]
    required: set[str]
    delegations: dict[str, str]  # role -> deputy, agreed earlier
    released_on: date

    @model_validator(mode="after")
    def complete_and_valid(self) -> ReleaseRecord:
        if self.gate_outcome not in {"pass",
                                     "pass-with-override"}:
            raise ValueError("only a passing gate is released")
        signed = set()
        for s in self.signatures:
            if s.on_behalf_of is None:
                signed.add(s.role)
            elif self.delegations.get(s.on_behalf_of) == \
                    s.person:
                signed.add(s.on_behalf_of)
            else:
                raise ValueError(
                    f"{s.person} is not a recorded deputy for "
                    f"{s.on_behalf_of}")
        missing = self.required - signed
        if missing:
            raise ValueError(f"unsigned: {sorted(missing)}")
        return self


class Level(IntEnum):
    AD_HOC = 1
    DEFINED = 2
    MEASURED = 3
    GATED = 4
    CONTINUOUS = 5


CRITERIA: dict[Level, set[str]] = {
    Level.DEFINED: {"plan", "owners", "versioned_datasets"},
    Level.MEASURED: {"error_analysis", "code_evaluators",
                     "validated_judges", "intervals"},
    Level.GATED: {"baselines", "change_detection",
                  "release_policy", "ci_gate"},
    Level.CONTINUOUS: {"tracing", "online_evaluation",
                       "drift_monitors", "triage_to_regression",
                       "release_records"},
}


def assess(capabilities: set[str]) -> Level:
    """Highest level whose criteria, and all below, are met."""
    level = Level.AD_HOC
    for lv in sorted(CRITERIA):
        if not CRITERIA[lv] <= capabilities:
            break
        level = lv
    return level
