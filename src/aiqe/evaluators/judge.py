"""aiqe/evaluators/judge.py: LLM judges behind the Provider."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from string import Template
from typing import Literal

from pydantic import BaseModel, ValidationError

from aiqe.datasets import EvalCase
from aiqe.evaluators import Evaluator, Score
from aiqe.providers import Provider
from aiqe.rubrics import Rubric


class JudgeConfig(BaseModel):
    """Everything that determines a judge's behaviour."""

    judge_id: str
    model: str            # a dated snapshot, never an alias
    template_path: str
    template_sha: str     # first 16 hex chars of SHA-256
    rubric_id: str
    rubric_version: str
    criterion_id: str


def load_template(path: Path) -> tuple[Template, str]:
    text = path.read_text(encoding="utf-8")
    sha = hashlib.sha256(text.encode()).hexdigest()[:16]
    return Template(text), sha


class JudgeVerdict(BaseModel):
    reasoning: str
    verdict: Literal[0, 1]


FENCE = re.compile(r"^`{3}(?:json)?\s*|\s*`{3}$")


def parse_verdict(raw: str) -> JudgeVerdict | None:
    """Parse strict JSON; None if the judge broke the format."""
    try:
        return JudgeVerdict.model_validate(
            json.loads(FENCE.sub("", raw.strip())))
    except (json.JSONDecodeError, ValidationError):
        return None


class SingleJudge:
    """Binary judge for one rubric criterion."""

    def __init__(self, provider: Provider, config: JudgeConfig,
                 rubric: Rubric) -> None:
        template, sha = load_template(Path(config.template_path))
        if sha != config.template_sha:
            raise ValueError(f"{config.judge_id}: template changed")
        if (rubric.id, rubric.version) != (
                config.rubric_id, config.rubric_version):
            raise ValueError(f"{config.judge_id}: rubric mismatch")
        self.provider = provider
        self.config = config
        self.template = template
        self.rubric = rubric
        self.name = config.judge_id

    def prompt(self, case: EvalCase, output: str) -> str:
        ref = case.reference.values if case.reference else []
        return self.template.substitute(
            rubric=self.rubric.render(), input=case.input,
            output=output, reference="\n".join(ref))

    def evaluate(self, case: EvalCase, output: str) -> Score:
        raw = self.provider.complete(
            self.config.model, self.prompt(case, output)).text
        v = parse_verdict(raw)
        if v is None:
            return Score(case_id=case.case_id, evaluator=self.name,
                         passed=None, detail="unparseable verdict")
        return Score(case_id=case.case_id, evaluator=self.name,
                     passed=v.verdict == 1, detail=v.reasoning)


class Cascade:
    """Use the first evaluator's verdict; defer to the second."""

    def __init__(self, name: str, first: Evaluator,
                 second: Evaluator) -> None:
        self.name = name
        self.first = first
        self.second = second

    def evaluate(self, case: EvalCase, output: str) -> Score:
        s = self.first.evaluate(case, output)
        if s.passed is None:
            s = self.second.evaluate(case, output)
        return s.model_copy(update={
            "evaluator": self.name,
            "detail": f"[{s.evaluator}] {s.detail}"})


class Preference(BaseModel):
    winner: Literal["A", "B"]
    reasoning: str


Outcome = Literal["A", "B", "inconsistent", "unparseable"]


class PairwiseJudge:
    """Asks in both orders to expose position bias."""

    def __init__(self, provider: Provider, model: str,
                 template: Template) -> None:
        self.provider = provider
        self.model = model
        self.template = template

    def _ask(self, case: EvalCase, first: str,
             second: str) -> str | None:
        raw = self.provider.complete(
            self.model, self.template.substitute(
                input=case.input, answer_a=first,
                answer_b=second)).text
        try:
            p = Preference.model_validate(
                json.loads(FENCE.sub("", raw.strip())))
        except (json.JSONDecodeError, ValidationError):
            return None
        return p.winner

    def compare(self, case: EvalCase, a: str, b: str) -> Outcome:
        ab = self._ask(case, a, b)
        ba = self._ask(case, b, a)
        if ab is None or ba is None:
            return "unparseable"
        swapped = {"A": "B", "B": "A"}[ba]
        return "A" if ab == swapped == "A" else (
            "B" if ab == swapped == "B" else "inconsistent")
