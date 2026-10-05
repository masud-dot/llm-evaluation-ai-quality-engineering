"""Online evaluation worker (batch mode).

Reads exported conversations as JSON lines, samples them,
applies reference-free evaluators, writes online scores, and
opens a triage item for every failure.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from aiqe.evaluators import Evaluator
from aiqe.evaluators.code import (CommitmentPreFilter,
                                  ContactAllowList)
from aiqe.online import OnlineScore, evaluate_sampled
from aiqe.tracing import sample
from aiqe.triage import TriageItem

RISKY = frozenset({"credit", "escalated", "guardrail"})


class Conversation(BaseModel):
    conversation_id: str
    customer_text: str
    answer: str
    tags: list[str] = []


def process(lines: list[str], base_rate: float,
            evaluators: list[Evaluator], now: datetime
            ) -> tuple[list[OnlineScore], list[TriageItem]]:
    scores: list[OnlineScore] = []
    items: list[TriageItem] = []
    for line in lines:
        if not line.strip():
            continue
        c = Conversation.model_validate_json(line)
        if not sample(c.conversation_id, c.tags, base_rate,
                      RISKY):
            continue
        got = evaluate_sampled(c.conversation_id, c.customer_text,
                               c.answer, ["OBJ-CREDIT-01",
                                          "OBJ-CONTACT-01"],
                               evaluators, now)
        scores += got
        items += [TriageItem(
            item_id=f"{c.conversation_id}-{s.score.evaluator}",
            conversation_id=c.conversation_id,
            signal=f"online {s.score.evaluator} failure")
            for s in got if s.score.passed is False]
    return scores, items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--triage", type=Path, required=True)
    ap.add_argument("--base-rate", type=float, default=0.05)
    a = ap.parse_args(argv)
    evs: list[Evaluator] = [
        CommitmentPreFilter(),
        ContactAllowList({"claims@tidewater-logistics.example"},
                         {"020 7946 0321"},
                         {"https://tidewater-logistics.example"
                          "/claims"})]
    scores, items = process(
        a.input.read_text(encoding="utf-8").splitlines(),
        a.base_rate, evs, datetime.now(timezone.utc))
    a.scores.write_text("".join(s.model_dump_json() + "\n"
                                for s in scores))
    a.triage.write_text("".join(i.model_dump_json() + "\n"
                                for i in items))
    return 0


if __name__ == "__main__":
    sys.exit(main())
