import shutil
import sys
from pathlib import Path

sys.path.insert(0, "projects/p5-production-platform")
from walkthrough import Carried, walkthrough  # noqa: E402

from aiqe.stats import paired_compare  # noqa: E402
from aiqe.triage import Status  # noqa: E402

SAME = {f"c{i}": True for i in range(60)}
# Carried from the last accepted gate run (constructed values).
CARRIED = Carried(
    values={"OBJ-CREDIT-01.failure_rate.upper": 0.0076,
            "OBJ-AGENT-01.failure_rate.upper": 0.012,
            "OBJ-AGENT-02.forbidden_attempts": 0,
            "OBJ-REL-01.pass3_mean.lower": 0.96,
            "OBJ-LAT-01.p95_seconds.upper": 3.2,
            "OBJ-CLAIM-01.failures": 0,
            "OBJ-SAFE-01.unsafe_responses": 0,
            "OBJ-SAFE-02.canary_leaks": 0,
            "OBJ-CONTACT-01.failures": 0,
            "OBJ-HELP-02.over_refusal_rate.upper": 0.06,
            "cost.per_task_gbp.mean": 0.015},
    paired={e: paired_compare(SAME, SAME)
            for e in ["credit-cascade", "claim-deadline",
                      "help-next-step"]})


def test_failure_travels_to_a_blocked_release(
        tmp_path: Path) -> None:
    for d in ["datasets", "gates"]:
        shutil.copytree(d, tmp_path / d)
    out = walkthrough(tmp_path, CARRIED)
    assert out.spans == 1 and out.sampled
    assert out.online_passed is False
    assert out.item is Status.PROMOTED
    assert out.versions == ["1.0", "1.1"]
    assert out.blocked == "fail"
    assert out.findings == ["REGRESSION-SET"]
    assert out.fixed == "pass"
