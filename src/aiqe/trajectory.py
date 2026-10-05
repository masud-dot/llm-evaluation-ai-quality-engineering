"""aiqe/trajectory.py: evaluating how an agent reached its end."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    args: dict[str, object]
    result: str = ""

    @property
    def error(self) -> bool:
        return self.result.startswith("error:")


class Trajectory(BaseModel):
    task_id: str
    calls: list[ToolCall]
    final_message: str


class ExpectedCall(BaseModel):
    """A call that should occur; only listed args are checked."""

    name: str
    args: dict[str, object] = Field(default_factory=dict)

    def matches(self, call: ToolCall) -> bool:
        return call.name == self.name and all(
            call.args.get(k) == v for k, v in self.args.items())


def exact(expected: Sequence[ExpectedCall],
          calls: Sequence[ToolCall]) -> bool:
    """Same calls, same order, nothing else."""
    return len(expected) == len(calls) and all(
        e.matches(c) for e, c in zip(expected, calls))


def in_order(expected: Sequence[ExpectedCall],
             calls: Sequence[ToolCall]) -> bool:
    """Expected calls appear in order; extra calls allowed."""
    it = iter(calls)
    return all(any(e.matches(c) for c in it) for e in expected)


def any_order(expected: Sequence[ExpectedCall],
              calls: Sequence[ToolCall]) -> bool:
    """Each expected call matched by a distinct actual call."""
    used: set[int] = set()
    for e in expected:
        hit = next((i for i, c in enumerate(calls)
                    if i not in used and e.matches(c)), None)
        if hit is None:
            return False
        used.add(hit)
    return True


class ProcessReport(BaseModel):
    task_id: str
    unexpected_tools: list[str]   # tools not allowed for task
    forbidden_attempts: int       # calls refused by policy
    identical_retries: int        # same call repeated after error
    steps: int
    over_budget: bool


def process_report(t: Trajectory, allowed: set[str],
                   step_budget: int) -> ProcessReport:
    unexpected = sorted({c.name for c in t.calls
                         if c.name not in allowed})
    forbidden = sum(c.result.startswith("error: not eligible")
                    for c in t.calls)
    retries = 0
    for prev, cur in zip(t.calls, t.calls[1:]):
        if prev.error and (prev.name, prev.args) == (
                cur.name, cur.args):
            retries += 1
    return ProcessReport(
        task_id=t.task_id, unexpected_tools=unexpected,
        forbidden_attempts=forbidden, identical_retries=retries,
        steps=len(t.calls),
        over_budget=len(t.calls) > step_budget)


def tool_selection(pairs: Sequence[tuple[str, str]]
                   ) -> Counter[tuple[str, str]]:
    """Confusion counts of (expected tool, first tool called)."""
    return Counter(pairs)
