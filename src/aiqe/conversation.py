"""aiqe/conversation.py: scripted multi-turn evaluation."""
from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Expectation(BaseModel):
    """Something that must hold at a given assistant turn."""

    turn: int                      # 0-based assistant turn
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)


class ConversationCase(BaseModel):
    case_id: str
    objective_ids: list[str] = Field(min_length=1)
    user_turns: list[str] = Field(min_length=1)
    expectations: list[Expectation]


System = Callable[[list[Message]], str]


def play(case: ConversationCase,
         system: System) -> list[Message]:
    """Feed scripted user turns; the system sees full history."""
    history: list[Message] = []
    for text in case.user_turns:
        history.append(Message(role="user", content=text))
        reply = system(list(history))
        history.append(Message(role="assistant", content=reply))
    return history


class TurnResult(BaseModel):
    turn: int
    passed: bool
    missing: list[str]
    forbidden: list[str]


def check(case: ConversationCase,
          transcript: list[Message]) -> list[TurnResult]:
    replies = [m.content for m in transcript
               if m.role == "assistant"]
    results = []
    for e in case.expectations:
        text = replies[e.turn].lower()
        missing = [s for s in e.must_include
                   if s.lower() not in text]
        forbidden = [s for s in e.must_not_include
                     if s.lower() in text]
        results.append(TurnResult(
            turn=e.turn, passed=not missing and not forbidden,
            missing=missing, forbidden=forbidden))
    return results


def render(history: list[Message]) -> str:
    """Canonical text of a history, for replay keys."""
    return "\n".join(f"{m.role}: {m.content}" for m in history)
