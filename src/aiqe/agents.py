"""aiqe/agents.py: outcome evaluation for stateful agents."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from fnmatch import fnmatch

from pydantic import BaseModel, Field

Snapshot = Mapping[str, object]      # flat path -> value


def diff(before: Snapshot, after: Snapshot) -> set[str]:
    """Paths added, removed, or changed."""
    keys = set(before) | set(after)
    return {k for k in keys if before.get(k) != after.get(k)}


class OutcomeCheck(BaseModel):
    name: str
    predicate: Callable[[Snapshot], bool]
    required: bool = True


class AgentTask(BaseModel):
    task_id: str
    objective_ids: list[str] = Field(min_length=1)
    instruction: str
    seed: str                          # named starting state
    checks: list[OutcomeCheck] = Field(min_length=1)
    may_change: list[str]              # glob patterns of paths


class OutcomeResult(BaseModel):
    task_id: str
    success: bool
    partial: float                     # share of checks passed
    failed_checks: list[str]
    side_effects: list[str]            # changes not permitted
    claimed_success: bool
    misreported: bool                  # claimed but not achieved


def evaluate_outcome(task: AgentTask, before: Snapshot,
                     after: Snapshot,
                     claimed_success: bool) -> OutcomeResult:
    results = {c.name: c.predicate(after) for c in task.checks}
    failed = [n for n, ok in results.items() if not ok]
    required_ok = all(results[c.name] for c in task.checks
                      if c.required)
    side = sorted(p for p in diff(before, after)
                  if not any(fnmatch(p, g)
                             for g in task.may_change))
    success = required_ok and not side
    return OutcomeResult(
        task_id=task.task_id, success=success,
        partial=sum(results.values()) / len(results),
        failed_checks=failed, side_effects=side,
        claimed_success=claimed_success,
        misreported=claimed_success and not success)
