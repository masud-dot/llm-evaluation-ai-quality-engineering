"""aiqe/strategy.py: quality objectives and the evaluation plan."""
from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class Dimension(StrEnum):
    CORRECTNESS = "correctness"
    GROUNDEDNESS = "groundedness"
    HELPFULNESS = "helpfulness"
    SAFETY = "safety"
    CONSISTENCY = "consistency"
    COST = "cost"
    LATENCY = "latency"
    RELIABILITY = "reliability"


class RiskTier(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    STANDARD = "standard"


class EvaluatorKind(StrEnum):
    CODE = "code"
    LLM_JUDGE = "llm_judge"
    HUMAN = "human"


class QualityObjective(BaseModel):
    id: str = Field(pattern=r"^OBJ-[A-Z]+-\d{2}$")
    dimension: Dimension
    statement: str
    risk: RiskTier
    owner: str
    evaluators: list[EvaluatorKind] = Field(min_length=1)
    acceptance: str
    evidence: list[str] = Field(default=["offline"])

    @model_validator(mode="after")
    def critical_needs_more_than_a_judge(
        self,
    ) -> QualityObjective:
        judge_only = self.evaluators == [EvaluatorKind.LLM_JUDGE]
        if self.risk is RiskTier.CRITICAL and judge_only:
            raise ValueError(
                f"{self.id}: a critical objective cannot rest "
                "on an LLM judge alone"
            )
        return self


class EvaluationPlan(BaseModel):
    system: str
    version: str
    objectives: list[QualityObjective]

    @model_validator(mode="after")
    def ids_are_unique(self) -> EvaluationPlan:
        ids = [o.id for o in self.objectives]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate ids: {sorted(dupes)}")
        return self

    def by_risk(self, tier: RiskTier) -> list[QualityObjective]:
        return [o for o in self.objectives if o.risk is tier]


def load_plan(path: Path) -> EvaluationPlan:
    with path.open("rb") as fh:
        return EvaluationPlan.model_validate(tomllib.load(fh))
