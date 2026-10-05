from aiqe.trajectory import (ExpectedCall, ToolCall, Trajectory,
                             any_order, exact, in_order,
                             process_report)
from systems.compass.sandbox import fresh

CLAIM = ToolCall(name="open_claim",
                 args={"shipment_id": "TW-2210",
                       "claim_type": "damage"},
                 result="CLM-0001")
REROUTE = ToolCall(name="reroute_shipment",
                   args={"shipment_id": "TW-1042",
                         "address_id": "ADDR-2"}, result="ok")
EXP = [ExpectedCall(name="reroute_shipment",
                    args={"shipment_id": "TW-1042"})]


def test_matchers() -> None:
    assert exact(EXP, [REROUTE])
    assert not exact(EXP, [CLAIM, REROUTE])
    assert in_order(EXP, [CLAIM, REROUTE])
    two = [ExpectedCall(name="open_claim"), EXP[0]]
    assert in_order(two, [CLAIM, REROUTE])
    assert not in_order(two, [REROUTE, CLAIM])
    assert any_order(two, [REROUTE, CLAIM])
    wrong = REROUTE.model_copy(
        update={"args": {"shipment_id": "TW-2210"}})
    assert not in_order(EXP, [wrong])


def test_process_report_from_sandbox() -> None:
    box = fresh("two-shipments")
    cid = box.open_claim("TW-2210", "damage")
    res = box.issue_credit(cid)
    credit = ToolCall(name="issue_credit",
                      args={"claim_id": cid}, result=res)
    t = Trajectory(task_id="t", final_message="m",
                   calls=[credit, credit, REROUTE])
    r = process_report(t, allowed={"issue_credit", "open_claim"},
                       step_budget=2)
    assert r.forbidden_attempts == 2
    assert r.identical_retries == 1
    assert r.unexpected_tools == ["reroute_shipment"]
    assert r.over_budget
