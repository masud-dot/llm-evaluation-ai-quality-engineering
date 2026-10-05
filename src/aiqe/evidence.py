"""aiqe/evidence.py: turn run results into gate evidence."""
from __future__ import annotations

from aiqe.metrics import Rate
from aiqe.policy import Evidence
from aiqe.stats import Paired, rate_interval


class EvidenceBuilder:
    """Names follow <objective>.<metric>.<bound>."""

    def __init__(self) -> None:
        self.values: dict[str, float] = {}
        self.paired: dict[str, Paired] = {}

    def failure_rate(self, objective: str,
                     rate: Rate) -> EvidenceBuilder:
        """Upper bound of the failure rate: 1 - lower pass."""
        iv = rate_interval(rate)
        self.values[f"{objective}.failure_rate.upper"] = (
            1 - iv.low)
        return self

    def count(self, name: str, n: int) -> EvidenceBuilder:
        self.values[name] = float(n)
        return self

    def value(self, name: str, v: float) -> EvidenceBuilder:
        self.values[name] = v
        return self

    def comparison(self, evaluator: str,
                   p: Paired) -> EvidenceBuilder:
        self.paired[evaluator] = p
        return self

    def build(self) -> Evidence:
        return Evidence(values=dict(self.values),
                        paired=dict(self.paired))
