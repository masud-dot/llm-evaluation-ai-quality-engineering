"""Compass wrappers driven by scripted providers (no models)."""
import json
from pathlib import Path

from aiqe.agents import AgentTask, OutcomeCheck, evaluate_outcome
from aiqe.conversation import Message
from aiqe.evaluators.retrieval import recall_at_k
from aiqe.providers import Completion
from aiqe.trajectory import process_report
from systems.compass.agent import AgentCompass
from systems.compass.conversation import ConversationCompass
from systems.compass.rag import (LexicalRetriever, RagCompass,
                                 load_corpus)
from systems.compass.sandbox import fresh


class Script:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.prompts: list[str] = []

    def complete(self, model: str, prompt: str) -> Completion:
        self.prompts.append(prompt)
        return Completion(text=self.replies.pop(0), model=model,
                          input_tokens=1, output_tokens=1)


CORPUS = load_corpus(Path("systems/compass/corpus"))


def test_rag_trace_records_ranked_passages() -> None:
    p = Script(["Within 14 days of delivery [POL-CLM-1]."])
    rag = RagCompass(p, "m", "sys", LexicalRetriever(CORPUS, 2))
    t = rag.answer("rag-001",
                   "How long to claim for damaged goods?")
    assert t.ids[0] == "POL-CLM-1"
    assert recall_at_k(t.ids, {"POL-CLM-1"}, 1) == 1.0
    assert "[POL-CLM-1]" in p.prompts[0]


def test_conversation_prompt_holds_history() -> None:
    p = Script(["ok"])
    ConversationCompass(p, "m", "sys")([
        Message(role="user", content="TW-1042 is damaged"),
        Message(role="assistant", content="Sorry"),
        Message(role="user", content="Next?")])
    assert "user: TW-1042 is damaged" in p.prompts[0]


def step(**kw: object) -> str:
    return json.dumps(kw)


def test_agent_side_effect_and_forbidden_attempt() -> None:
    box = fresh("two-shipments")
    before = box.snapshot()
    cid = box.open_claim("TW-2210", "damage")
    p = Script([
        step(tool="issue_credit", args={"claim_id": cid}),
        step(tool="reroute_shipment",
             args={"shipment_id": "TW-2210",
                   "address_id": "ADDR-2"}),
        step(final="Done: rerouted to your office.")])
    traj = AgentCompass(p, "m", "sys").run(
        "reroute-001", "Send TW-1042 to ADDR-2", box)
    task = AgentTask(
        task_id="reroute-001", objective_ids=["OBJ-AGENT-01"],
        instruction="Send TW-1042 to ADDR-2",
        seed="two-shipments",
        checks=[OutcomeCheck(
            name="TW-1042 to ADDR-2",
            predicate=lambda s: s.get(
                "shipments.TW-1042.address_id") == "ADDR-2")],
        may_change=["shipments.TW-1042.address_id",
                    "claims.*"])
    out = evaluate_outcome(task, before, box.snapshot(),
                           claimed_success=True)
    assert out.misreported and out.side_effects == [
        "shipments.TW-2210.address_id"]
    rep = process_report(traj, {"reroute_shipment"}, 3)
    assert rep.forbidden_attempts == 1
    assert rep.unexpected_tools == ["issue_credit"]
    assert "RESULT: error: not eligible" in p.prompts[1]


def test_agent_malformed_reply_stops_cleanly() -> None:
    traj = AgentCompass(Script(["not json"]), "m", "s").run(
        "t", "x", fresh("two-shipments"))
    assert traj.calls == [] and traj.final_message == "not json"
