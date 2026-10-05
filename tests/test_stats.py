import pytest

from aiqe.metrics import Rate
from aiqe.stats import (cases_for_zero_failures, case_bootstrap,
                        holm, paired_compare, pass_at_k,
                        pass_hat_k, rate_interval, wilson)


def test_wilson_and_rate_interval() -> None:
    a, b = wilson(49, 60), wilson(51, 60)
    assert a.low < b.estimate < a.high      # overlap
    r = Rate(evaluator="e", passed=49, failed=11, deferred=40)
    assert rate_interval(r) == a
    z = wilson(80, 80)
    assert z.estimate == 1.0 and 1 - z.low > 0.04


def test_zero_failure_sample_size() -> None:
    n = cases_for_zero_failures(0.01)
    assert n == 381
    assert 1 - wilson(n, n).low <= 0.01
    assert 1 - wilson(n - 1, n - 1).low > 0.01


def test_case_bootstrap_groups_trials() -> None:
    trials = {f"c{i}": [True, True, False] for i in range(30)}
    iv = case_bootstrap(trials, seed=1)
    assert iv.estimate == pytest.approx(2 / 3)
    assert iv.low == iv.high == pytest.approx(2 / 3)
    assert case_bootstrap(trials, seed=1) == iv


def test_paired_compare_constructed() -> None:
    base = {f"c{i}": True for i in range(60)}
    cand = dict(base)
    for i in range(11):
        base[f"c{i}"] = False
    for i in range(7):
        cand[f"c{i}"] = False
    for i in range(11, 13):          # candidate-only failures
        cand[f"c{i}"] = False
    res = paired_compare(base, cand, seed=3)
    assert (res.baseline_only, res.candidate_only) == (2, 4)
    assert res.difference.estimate == pytest.approx(2 / 60)
    assert res.difference.low < 0 < res.difference.high
    assert res.p_value == pytest.approx(0.6875)


def test_holm() -> None:
    assert holm([0.001, 0.04, 0.03]) == [True, False, False]


def test_pass_at_k_and_pass_hat_k() -> None:
    assert pass_at_k(10, 9, 3) == 1.0
    assert pass_hat_k(10, 9, 3) == pytest.approx(0.7)
    assert pass_at_k(10, 5, 1) == pytest.approx(0.5)
    assert pass_hat_k(10, 10, 5) == 1.0
