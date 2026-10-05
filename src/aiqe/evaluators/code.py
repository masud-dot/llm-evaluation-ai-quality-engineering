"""aiqe/evaluators/code.py: deterministic evaluators."""
from __future__ import annotations

import re

from aiqe.datasets import EvalCase
from aiqe.evaluators import Score

WORDS = {"seven": 7, "fourteen": 14, "thirty": 30}
DURATION = re.compile(r"\b(\d+|seven|fourteen|thirty)\s+days?\b",
                      re.IGNORECASE)
START = re.compile(
    r"\b(?:from|of|after)\s+(?:the\s+|your\s+)?"
    r"(expected\s+delivery|order|dispatch|delivery)",
    re.IGNORECASE)

# Tidewater claims policy: claim type -> (days, start event)
CLAIM_POLICY = {"damage": (14, "delivery"),
                "loss": (30, "expected delivery")}


def _days(text: str) -> list[int]:
    out = []
    for m in DURATION.finditer(text):
        tok = m.group(1).lower()
        out.append(int(tok) if tok.isdigit() else WORDS[tok])
    return out


class ClaimDeadline:
    """OBJ-CLAIM-01: duration and starting event match policy."""

    name = "claim-deadline"

    def evaluate(self, case: EvalCase, output: str) -> Score:
        kind = next((t for t in case.tags
                     if t in CLAIM_POLICY), None)
        if kind is None:
            return self._score(case, None, "not a claim case")
        days, start = CLAIM_POLICY[kind]
        stated = _days(output)
        if not stated:
            return self._score(case, False, "no deadline stated")
        if any(d != days for d in stated):
            return self._score(case, False,
                               f"duration {stated} != {days}")
        m = START.search(output)
        if m:
            said = " ".join(m.group(1).lower().split())
            if said != start:
                return self._score(
                    case, False, f"starts at '{said}'")
        return self._score(case, True, "")

    def _score(self, case: EvalCase, passed: bool | None,
               detail: str) -> Score:
        return Score(case_id=case.case_id, evaluator=self.name,
                     passed=passed, detail=detail)


EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
PHONE = re.compile(r"(?:\+44\s?|0)\d{2,4}(?:[\s-]?\d{3,4}){2}")


def _norm_phone(p: str) -> str:
    digits = re.sub(r"\D", "", p)
    return "0" + digits[2:] if digits.startswith("44") else digits


class ContactAllowList:
    """OBJ-CONTACT-01: every contact detail is in the directory."""

    name = "contact-allow-list"

    def __init__(self, emails: set[str], phones: set[str],
                 urls: set[str]) -> None:
        self.emails = {e.lower() for e in emails}
        self.phones = {_norm_phone(p) for p in phones}
        self.urls = {u.lower().rstrip("/") for u in urls}

    def evaluate(self, case: EvalCase, output: str) -> Score:
        bad = [e for e in EMAIL.findall(output)
               if e.lower() not in self.emails]
        bad += [p for p in PHONE.findall(output)
                if _norm_phone(p) not in self.phones]
        bad += [u for u in URL.findall(output)
                if u.lower().rstrip("/.,") not in self.urls]
        return Score(case_id=case.case_id, evaluator=self.name,
                     passed=not bad,
                     detail=f"not in directory: {bad}" if bad
                     else "")


COMMITMENT = re.compile(
    r"\b(?:you(?:'ll| will)|we(?:'ll| will))\s+"
    r"(?:receive|get|be given|give you|issue|refund)\b",
    re.IGNORECASE)


class CommitmentPreFilter:
    """FM-CREDIT-PROMISE pre-filter.

    Fails unambiguous commitments; defers everything else,
    because absence of a phrase does not prove absence of a
    commitment.
    """

    name = "commitment-pre-filter"

    def evaluate(self, case: EvalCase, output: str) -> Score:
        m = COMMITMENT.search(output)
        return Score(case_id=case.case_id, evaluator=self.name,
                     passed=False if m else None,
                     detail=f"matched '{m.group(0)}'" if m
                     else "deferred")


def same_deadline(outputs: list[str]) -> bool:
    """Metamorphic relation: paraphrases of one claim question
    must state the same set of durations."""
    stated = [frozenset(_days(o)) for o in outputs]
    return len(set(stated)) == 1
