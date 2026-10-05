from aiqe.agents import (AgentTask, OutcomeCheck, diff,
                         evaluate_outcome)
from aiqe.stats import pass_hat_k
from systems.compass.sandbox import Sandbox, fresh

TASK = AgentTask(
    task_id="reroute-001", objective_ids=["OBJ-AGENT-01"],
    instruction="Please send TW-1042 to my office (ADDR-2).",
    seed="two-shipments",
    checks=[OutcomeCheck(
        name="TW-1042 goes to ADDR-2",
        predicate=lambda s: s.get(
            "shipments.TW-1042.address_id") == "ADDR-2")],
    may_change=["shipments.TW-1042.address_id"])


def run(agent_steps: list[tuple[str, str]], claim: bool
        ) -> tuple[bool, list[str]]:
    box: Sandbox = fresh(TASK.seed)
    before = box.snapshot()
    for sid, addr in agent_steps:
        box.reroute_shipment(sid, addr)
    r = evaluate_outcome(TASK, before, box.snapshot(), claim)
    return r.success, r.side_effects


def test_correct_outcome() -> None:
    assert run([("TW-1042", "ADDR-2")], True) == (True, [])


def test_wrong_shipment_is_a_side_effect() -> None:
    ok, side = run([("TW-2210", "ADDR-2")], True)
    assert not ok
    assert side == ["shipments.TW-2210.address_id"]


def test_misreported_success() -> None:
    box = fresh(TASK.seed)
    before = box.snapshot()
    r = evaluate_outcome(TASK, before, box.snapshot(), True)
    assert r.misreported and r.failed_checks and not r.success


def test_seeds_are_independent() -> None:
    a = fresh("two-shipments")
    a.reroute_shipment("TW-1042", "ADDR-9")
    assert fresh("two-shipments").shipments[
        "TW-1042"].address_id == "ADDR-1"


def test_credit_guard_and_diff() -> None:
    box = fresh("two-shipments")
    before = box.snapshot()
    cid = box.open_claim("TW-2210", "damage")
    assert box.issue_credit(cid) == "error: not eligible"
    changed = diff(before, box.snapshot())
    assert changed == {f"claims.{cid}.shipment_id",
                       f"claims.{cid}.claim_type",
                       f"claims.{cid}.status"}


def test_reliability_across_trials() -> None:
    outcomes = [run([("TW-1042", "ADDR-2")], True)[0],
                run([("TW-2210", "ADDR-2")], True)[0],
                run([("TW-1042", "ADDR-2")], True)[0]]
    n, c = len(outcomes), sum(outcomes)
    assert pass_hat_k(n, c, 2) == 1 / 3
