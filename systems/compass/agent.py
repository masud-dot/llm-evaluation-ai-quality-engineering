"""systems/compass/agent.py: Compass acting through tools.

The model replies with JSON: {"tool": name, "args": {...}} to
act, or {"final": text} to finish. Every step, tool call, and
result is recorded, and the prompt carries the full transcript,
so replay keys change whenever an earlier step changes.
"""
from __future__ import annotations

import json
from collections.abc import Callable

from pydantic import BaseModel, ValidationError

from aiqe.providers import Provider
from aiqe.trajectory import ToolCall, Trajectory
from systems.compass.sandbox import Sandbox


class Step(BaseModel):
    tool: str | None = None
    args: dict[str, str] = {}
    final: str | None = None


TOOLS = ("reroute_shipment", "open_claim", "issue_credit")


class AgentCompass:
    def __init__(self, provider: Provider, model: str,
                 system_prompt: str, max_steps: int = 8) -> None:
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    def run(self, task_id: str, instruction: str,
            box: Sandbox) -> Trajectory:
        calls: list[ToolCall] = []
        transcript = [f"CUSTOMER: {instruction}"]
        for _ in range(self.max_steps):
            prompt = "\n".join([self.system_prompt,
                                f"TOOLS: {', '.join(TOOLS)}",
                                *transcript])
            raw = self.provider.complete(self.model, prompt).text
            try:
                step = Step.model_validate(json.loads(raw))
            except (json.JSONDecodeError, ValidationError):
                return Trajectory(task_id=task_id, calls=calls,
                                  final_message=raw)
            if step.final is not None or step.tool is None:
                return Trajectory(task_id=task_id, calls=calls,
                                  final_message=step.final or "")
            result = self._execute(box, step.tool, step.args)
            calls.append(ToolCall(name=step.tool,
                                  args=dict(step.args),
                                  result=result))
            transcript += [f"CALL: {raw}", f"RESULT: {result}"]
        return Trajectory(task_id=task_id, calls=calls,
                          final_message="error: step limit")

    @staticmethod
    def _execute(box: Sandbox, name: str,
                 args: dict[str, str]) -> str:
        if name not in TOOLS:
            return f"error: unknown tool {name}"
        fn: Callable[..., str] = getattr(box, name)
        try:
            return fn(**args)
        except (KeyError, TypeError) as exc:
            return f"error: {type(exc).__name__}"
