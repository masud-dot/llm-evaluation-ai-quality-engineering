"""aiqe/stats.py: uncertainty for evaluation results.

Conventions: two-sided 95% Wilson intervals for proportions;
paired comparisons on the same cases; resampling by case.
"""
from __future__ import annotations

from collections.abc import Sequence
from math import comb, isfinite

import numpy as np
from pydantic import BaseModel
from scipy.stats import binomtest
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from aiqe.metrics import Rate


class Interval(BaseModel):
    estimate: float
    low: float
    high: float
    n: int


def wilson(passed: int, n: int,
           confidence: float = 0.95) -> Interval:
    ci = binomtest(passed, n).proportion_ci(
        confidence_level=confidence, method="wilson")
    return Interval(estimate=passed / n, low=ci.low,
                    high=ci.high, n=n)


def rate_interval(rate: Rate,
                  confidence: float = 0.95) -> Interval:
    """Interval for a Rate over its decided cases."""
    if rate.decided == 0:
        raise ValueError(f"{rate.evaluator}: nothing decided")
    return wilson(rate.passed, rate.decided, confidence)


def cases_for_zero_failures(max_rate: float,
                            confidence: float = 0.95) -> int:
    """Smallest n whose Wilson upper bound at 0 failures is at
    or below max_rate."""
    n = 1
    while wilson(n, n, confidence).low < 1 - max_rate:
        n += 1
    return n


Trials = dict[str, list[bool]]   # case -> verdict per trial


def case_bootstrap(trials: Trials, resamples: int = 2000,
                   seed: int = 0,
                   confidence: float = 0.95) -> Interval:
    """Mean pass rate with an interval, resampling whole cases
    so that repeated trials of one case stay together."""
    per_case = np.array([np.mean(v) for v in trials.values()])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_case),
                       (resamples, len(per_case)))
    means = per_case[idx].mean(axis=1)
    tail = (1 - confidence) / 2
    return Interval(estimate=float(per_case.mean()),
                    low=float(np.quantile(means, tail)),
                    high=float(np.quantile(means, 1 - tail)),
                    n=len(per_case))


class Paired(BaseModel):
    n: int
    baseline_only: int     # baseline passed, candidate failed
    candidate_only: int    # candidate passed, baseline failed
    difference: Interval   # candidate minus baseline pass rate
    p_value: float         # exact McNemar test


def paired_compare(baseline: dict[str, bool],
                   candidate: dict[str, bool],
                   resamples: int = 2000,
                   seed: int = 0) -> Paired:
    """Compare two versions on the cases both evaluated."""
    shared = sorted(set(baseline) & set(candidate))
    b = np.array([baseline[c] for c in shared], dtype=int)
    c = np.array([candidate[c] for c in shared], dtype=int)
    only_b = int(((b == 1) & (c == 0)).sum())
    only_c = int(((b == 0) & (c == 1)).sum())
    table = [[int(((b == 1) & (c == 1)).sum()), only_b],
             [only_c, int(((b == 0) & (c == 0)).sum())]]
    p = float(mcnemar(table, exact=True).pvalue)
    diff = c - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (resamples, len(diff)))
    boots = diff[idx].mean(axis=1)
    interval = Interval(
        estimate=float(diff.mean()),
        low=float(np.quantile(boots, 0.025)),
        high=float(np.quantile(boots, 0.975)), n=len(shared))
    return Paired(n=len(shared), baseline_only=only_b,
                  candidate_only=only_c, difference=interval,
                  p_value=p if isfinite(p) else 1.0)


def holm(p_values: Sequence[float],
         alpha: float = 0.05) -> list[bool]:
    """Which of several tests remain significant (Holm)."""
    reject = multipletests(list(p_values), alpha=alpha,
                           method="holm")[0]
    return [bool(r) for r in reject]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Chance at least one of k draws (from n trials with c
    passes) passes."""
    if n - c < k:
        return 1.0
    return 1 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k: chance all k draws (from n trials with c passes)
    pass."""
    return comb(c, k) / comb(n, k)
