from pathlib import Path

import pytest

from aiqe.cost import (Candidate, Direction, Price,
                       pareto_front, percentile_interval,
                       task_cost)
from aiqe.policy import Evidence, Verdict, apply, load_policy
from aiqe.providers import Completion
from aiqe.stats import paired_compare

POLICY = load_policy(Path("gates/compass-release.toml"))
GOOD = {"OBJ-CREDIT-01.failure_rate.upper": 0.008,
        "OBJ-AGENT-01.failure_rate.upper": 0.015,
        "OBJ-AGENT-02.forbidden_attempts": 0,
        "OBJ-REL-01.pass3_mean.lower": 0.96,
        "OBJ-LAT-01.p95_seconds.upper": 3.2,
        "cost.per_task_gbp.mean": 0.015,
        "OBJ-CLAIM-01.failures": 0,
        "OBJ-SAFE-01.unsafe_responses": 0,
        "OBJ-SAFE-02.canary_leaks": 0,
        "OBJ-CONTACT-01.failures": 0,
        "OBJ-HELP-02.over_refusal_rate.upper": 0.06,
        "regression.failures": 0}
SAME = {f"c{i}": True for i in range(50)}
NEUTRAL = {e: paired_compare(SAME, SAME)
           for e in ["credit-cascade", "claim-deadline",
                     "help-next-step"]}


def test_cost_with_illustrative_prices() -> None:
    prices = {"m": Price(input=1.0, output=4.0)}  # not real
    calls = [Completion(text="", model="m", input_tokens=1000,
                        output_tokens=250)] * 2
    assert task_cost(calls, prices) == pytest.approx(0.004)


def test_percentile_interval() -> None:
    iv = percentile_interval([1.0] * 95 + [9.0] * 5, 95)
    assert iv.low <= iv.estimate <= iv.high


def test_pareto() -> None:
    d: dict[str, Direction] = {"quality": "max",
                               "cost": "min"}
    cs = [Candidate(name="a", metrics={"quality": .9,
                                       "cost": 2.0}),
          Candidate(name="b", metrics={"quality": .8,
                                       "cost": 1.0}),
          Candidate(name="c", metrics={"quality": .8,
                                       "cost": 1.5})]
    assert pareto_front(cs, d) == ["a", "b"]


def test_policy_pass_review_fail() -> None:
    assert apply(POLICY, Evidence(values=GOOD,
                                  paired=NEUTRAL)).verdict \
        is Verdict.PASS
    pricey = dict(GOOD, **{"cost.per_task_gbp.mean": 0.03})
    r = apply(POLICY, Evidence(values=pricey, paired=NEUTRAL))
    assert r.verdict is Verdict.REVIEW
    flaky = dict(GOOD, **{"OBJ-REL-01.pass3_mean.lower": 0.90})
    r = apply(POLICY, Evidence(values=flaky, paired=NEUTRAL))
    assert r.verdict is Verdict.FAIL
    assert r.findings[0].objective == "OBJ-REL-01"


def test_missing_evidence_is_not_a_pass() -> None:
    partial = {k: v for k, v in GOOD.items()
               if not k.startswith("OBJ-LAT-01")}
    r = apply(POLICY, Evidence(values=partial, paired=NEUTRAL))
    assert r.verdict is Verdict.FAIL
    assert "no evidence" in r.findings[0].message
