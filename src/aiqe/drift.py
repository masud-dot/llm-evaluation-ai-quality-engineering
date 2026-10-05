"""aiqe/drift.py: input, output-quality, and evaluator drift."""
from __future__ import annotations

import math
from collections.abc import Mapping

from pydantic import BaseModel
from scipy.stats import chi2_contingency, fisher_exact

from aiqe.metrics import Rate
from aiqe.validation import Confusion


class InputDrift(BaseModel):
    psi: float
    p_value: float
    new_categories: list[str]
    growing: list[tuple[str, float, float]]  # name, before, now


def input_drift(baseline: Mapping[str, int],
                current: Mapping[str, int],
                floor: float = 1e-4) -> InputDrift:
    """Compare category mixes, e.g. routed intents per week."""
    cats = sorted(set(baseline) | set(current))
    b = [baseline.get(c, 0) for c in cats]
    c = [current.get(c, 0) for c in cats]
    bn, cn = sum(b), sum(c)
    psi = 0.0
    growing = []
    for name, x, y in zip(cats, b, c):
        p, q = max(x / bn, floor), max(y / cn, floor)
        psi += (q - p) * math.log(q / p)
        if q > 2 * p and y >= 5:
            growing.append((name, round(x / bn, 4),
                            round(y / cn, 4)))
    table = [[x + 0.5 for x in b], [y + 0.5 for y in c]]
    p_value = float(chi2_contingency(table)[1])
    return InputDrift(
        psi=psi, p_value=p_value,
        new_categories=[n for n, x in zip(cats, b) if x == 0],
        growing=growing)


def quality_shift(baseline: Rate, current: Rate) -> float:
    """Exact p-value that the pass rate differs between two
    windows of the same stratum."""
    table = [[baseline.passed, baseline.failed],
             [current.passed, current.failed]]
    return float(fisher_exact(table)[1])


def judge_drift(validated: Confusion,
                recent: Confusion) -> tuple[float, float]:
    """Change in TNR since validation, and its exact p-value."""
    table = [[validated.tn, validated.fp],
             [recent.tn, recent.fp]]
    return recent.tnr - validated.tnr, float(
        fisher_exact(table)[1])
