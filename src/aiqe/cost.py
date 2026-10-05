"""aiqe/cost.py: cost, latency, and Pareto trade-offs."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel

from aiqe.providers import Completion
from aiqe.stats import Interval


class Price(BaseModel):
    """Per million tokens, from a dated provider price list."""

    input: float
    output: float


def task_cost(calls: Sequence[Completion],
              prices: dict[str, Price]) -> float:
    """Cost of one task: every model call it made."""
    total = 0.0
    for c in calls:
        p = prices[c.model]
        total += (c.input_tokens * p.input
                  + c.output_tokens * p.output) / 1_000_000
    return total


def percentile_interval(values: Sequence[float], q: float,
                        resamples: int = 2000,
                        seed: int = 0) -> Interval:
    """Bootstrap interval for a latency percentile."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (resamples, len(v)))
    boots = np.percentile(v[idx], q, axis=1)
    return Interval(estimate=float(np.percentile(v, q)),
                    low=float(np.quantile(boots, 0.025)),
                    high=float(np.quantile(boots, 0.975)),
                    n=len(v))


Direction = Literal["max", "min"]


class Candidate(BaseModel):
    name: str
    metrics: dict[str, float]


def dominates(a: Candidate, b: Candidate,
              directions: dict[str, Direction]) -> bool:
    """a is at least as good everywhere and better somewhere."""
    better = False
    for m, d in directions.items():
        x, y = a.metrics[m], b.metrics[m]
        if (x < y) if d == "max" else (x > y):
            return False
        if x != y:
            better = True
    return better


def pareto_front(cands: Sequence[Candidate],
                 directions: dict[str, Direction]
                 ) -> list[str]:
    return [c.name for c in cands
            if not any(dominates(o, c, directions)
                       for o in cands if o is not c)]
