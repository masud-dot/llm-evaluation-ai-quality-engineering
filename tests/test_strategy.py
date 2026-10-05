from pathlib import Path

import pytest
from pydantic import ValidationError

from aiqe.strategy import (Dimension, EvaluatorKind,
                           QualityObjective, RiskTier,
                           load_plan)


def test_compass_plan_loads() -> None:
    plan = load_plan(Path("plans/compass-plan.toml"))
    crit = plan.by_risk(RiskTier.CRITICAL)
    assert [o.id for o in crit][:3] == [
        "OBJ-CREDIT-01", "OBJ-CLAIM-01", "OBJ-SAFE-01"]
    assert "\n" not in plan.objectives[0].statement


def test_critical_judge_only_rejected() -> None:
    with pytest.raises(ValidationError):
        QualityObjective(
            id="OBJ-CREDIT-02", dimension=Dimension.CORRECTNESS,
            statement="x", risk=RiskTier.CRITICAL, owner="y",
            evaluators=[EvaluatorKind.LLM_JUDGE],
            acceptance="z")
