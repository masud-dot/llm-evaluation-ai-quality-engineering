"""aiqe/evaluators/retrieval.py: evaluating RAG as a system."""
from __future__ import annotations

import math
import re
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from aiqe.datasets import Dataset, EvalCase
from aiqe.evaluators import Score
from aiqe.evaluators.judge import (JudgeConfig, load_template,
                                   parse_verdict)
from aiqe.providers import Provider


class RagCase(EvalCase):
    """An evaluation case that knows which passages answer it."""

    relevant_ids: list[str] = Field(min_length=1)


class Passage(BaseModel):
    id: str
    text: str


class RagTrace(BaseModel):
    case_id: str
    query: str
    retrieved: list[Passage]     # in rank order
    answer: str

    @property
    def ids(self) -> list[str]:
        return [p.id for p in self.retrieved]


def load_rag_jsonl(path: Path, name: str,
                   version: str) -> Dataset:
    lines = path.read_text(encoding="utf-8").splitlines()
    cases: list[EvalCase] = [RagCase.model_validate_json(x)
                             for x in lines if x.strip()]
    return Dataset(name=name, version=version, cases=cases)


def recall_at_k(ids: list[str], relevant: set[str],
                k: int) -> float:
    return len(set(ids[:k]) & relevant) / len(relevant)


def precision_at_k(ids: list[str], relevant: set[str],
                   k: int) -> float:
    top = ids[:k]
    return len(set(top) & relevant) / len(top) if top else 0.0


def reciprocal_rank(ids: list[str], relevant: set[str]) -> float:
    for rank, pid in enumerate(ids, start=1):
        if pid in relevant:
            return 1 / rank
    return 0.0


def ndcg_at_k(ids: list[str], gains: dict[str, float],
              k: int) -> float:
    """Graded relevance: gains maps passage id to relevance."""
    dcg = sum(gains.get(pid, 0.0) / math.log2(i + 2)
              for i, pid in enumerate(ids[:k]))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


class Attribution(StrEnum):
    OK = "ok"                       # found and answered
    RETRIEVAL = "retrieval"         # not found, answer wrong
    GENERATION = "generation"       # found, answer wrong
    UNGROUNDED = "ungrounded"       # not found, answer right


def attribute(found: bool, answer_ok: bool) -> Attribution:
    if found:
        return Attribution.OK if answer_ok else \
            Attribution.GENERATION
    return Attribution.UNGROUNDED if answer_ok else \
        Attribution.RETRIEVAL


CITE = re.compile(r"\[(POL-[A-Z]+-\d+)\]")


def citation_validity(trace: RagTrace) -> Score:
    """Invariant: every cited passage was actually retrieved."""
    cited = CITE.findall(trace.answer)
    bad = sorted(set(cited) - set(trace.ids))
    if not cited:
        return Score(case_id=trace.case_id,
                     evaluator="citation-validity", passed=None,
                     detail="no citations")
    return Score(case_id=trace.case_id,
                 evaluator="citation-validity", passed=not bad,
                 detail=f"not retrieved: {bad}" if bad else "")


class GroundednessJudge:
    """Reference-guided judge: is the answer supported by the
    passages that were retrieved for it?"""

    def __init__(self, provider: Provider,
                 config: JudgeConfig) -> None:
        template, sha = load_template(Path(config.template_path))
        if sha != config.template_sha:
            raise ValueError(f"{config.judge_id}: template changed")
        self.provider = provider
        self.config = config
        self.template = template
        self.name = config.judge_id

    def evaluate_trace(self, trace: RagTrace) -> Score:
        context = "\n\n".join(f"[{p.id}] {p.text}"
                              for p in trace.retrieved)
        prompt = self.template.substitute(
            input=trace.query, context=context,
            output=trace.answer)
        raw = self.provider.complete(self.config.model,
                                     prompt).text
        v = parse_verdict(raw)
        if v is None:
            return Score(case_id=trace.case_id,
                         evaluator=self.name, passed=None,
                         detail="unparseable verdict")
        return Score(case_id=trace.case_id, evaluator=self.name,
                     passed=v.verdict == 1, detail=v.reasoning)
