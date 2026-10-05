"""Comparison only. Runs in the adapters environment with
DEEPEVAL_TELEMETRY_OPT_OUT=YES."""
from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class ClaimDeadlineMetric(BaseMetric):
    """Wraps the aiqe idea of a code evaluator for DeepEval."""

    def __init__(self) -> None:
        self.threshold = 1.0

    def measure(self, test_case: LLMTestCase, *args: object,
                **kwargs: object) -> float:
        out = test_case.actual_output or ""
        score = 1.0 if "14 days" in out else 0.0
        self.score = score
        self.success = score >= 1.0
        self.reason = "deadline stated" if self.success else (
            "deadline missing or wrong")
        return score

    async def a_measure(self, test_case: LLMTestCase,
                        *args: object, **kwargs: object) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "ClaimDeadline"


def test_damage_deadline() -> None:
    case = LLMTestCase(
        input="How long do I have to report damage?",
        actual_output="Within 14 days of delivery.")
    assert_test(case, [ClaimDeadlineMetric()], run_async=False)
