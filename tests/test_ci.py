import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from aiqe.ci import Override, main, resolve
from aiqe.evidence import EvidenceBuilder
from aiqe.metrics import Rate
from aiqe.policy import apply, load_policy
from aiqe.stats import paired_compare

POLICY = Path("gates/compass-release.toml")
SAME = {f"c{i}": True for i in range(50)}


def evidence(cost: float = 0.015,
             leaks: int = 0) -> EvidenceBuilder:
    b = EvidenceBuilder()
    b.failure_rate("OBJ-CREDIT-01", Rate(
        evaluator="credit-cascade", passed=500, failed=0,
        deferred=0))
    b.failure_rate("OBJ-AGENT-01", Rate(
        evaluator="agent-outcome", passed=300, failed=0,
        deferred=0))
    for name in ["OBJ-AGENT-02.forbidden_attempts",
                 "OBJ-CLAIM-01.failures",
                 "OBJ-SAFE-01.unsafe_responses",
                 "OBJ-CONTACT-01.failures",
                 "regression.failures"]:
        b.count(name, 0)
    b.count("OBJ-SAFE-02.canary_leaks", leaks)
    b.value("OBJ-REL-01.pass3_mean.lower", 0.97)
    b.value("OBJ-LAT-01.p95_seconds.upper", 3.1)
    b.value("OBJ-HELP-02.over_refusal_rate.upper", 0.06)
    b.value("cost.per_task_gbp.mean", cost)
    for e in ["credit-cascade", "claim-deadline",
              "help-next-step"]:
        b.comparison(e, paired_compare(SAME, SAME))
    return b


def test_builder_uses_upper_bound() -> None:
    ev = evidence().build()
    v = ev.values["OBJ-CREDIT-01.failure_rate.upper"]
    assert 0 < v < 0.01          # 0/500 still has an upper bound


def test_resolve_rules() -> None:
    pol = load_policy(POLICY)
    today = date(2026, 9, 19)
    pricey = apply(pol, evidence(cost=0.03).build())
    ok = Override(objective="BUDGET-COST", approver="ops",
                  rationale="seasonal peak",
                  expires=today + timedelta(days=30))
    assert resolve(pricey, [], today)[0] == "review"
    assert resolve(pricey, [ok], today)[0] == \
        "pass-with-override"
    old = ok.model_copy(update={"expires": today
                                - timedelta(days=1)})
    assert resolve(pricey, [old], today)[0] == "review"
    leak = apply(pol, evidence(leaks=1).build())
    blk = ok.model_copy(update={"objective": "OBJ-SAFE-02"})
    assert resolve(leak, [blk], today)[0] == "fail"


def test_cli_gate_and_plan(tmp_path: Path) -> None:
    ev = tmp_path / "ev.json"
    ev.write_text(evidence().build().model_dump_json())
    (tmp_path / "ovr").mkdir()
    rc = main(["gate", "--policy", str(POLICY), "--evidence",
               str(ev), "--overrides", str(tmp_path / "ovr"),
               "--report", str(tmp_path / "r.md")])
    assert rc == 0
    assert "Quality gate: PASS" in (tmp_path / "r.md").read_text()
    base = {"prompt": "p1", "system_model": "m1",
            "provider_sdk": "s", "retriever": "r", "corpus": "c",
            "tools": "t", "judge_model": "j",
            "judge_template": "jt", "rubric": "1.0",
            "dataset": "d", "sandbox": "sb"}
    head = dict(base, judge_model="j2")
    (tmp_path / "b.json").write_text(json.dumps(base))
    (tmp_path / "h.json").write_text(json.dumps(head))
    assert main(["plan", "--base", str(tmp_path / "b.json"),
                 "--head", str(tmp_path / "h.json"),
                 "--out", str(tmp_path / "i.json")]) == 0
    imp = json.loads((tmp_path / "i.json").read_text())
    assert imp["rerecord_outputs"] is False
    assert sorted(imp["suites"]) == ["judge-validation",
                                     "regression"]
