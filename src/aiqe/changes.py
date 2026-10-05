"""aiqe/changes.py: detect changes and route them to suites."""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from aiqe.datasets import EvalCase, Reference, Source, Split


class Change(StrEnum):
    PROMPT = "prompt"
    SYSTEM_MODEL = "system_model"
    PROVIDER_SDK = "provider_sdk"
    RETRIEVER = "retriever"
    CORPUS = "corpus"
    TOOLS = "tools"                # code or descriptions
    JUDGE_MODEL = "judge_model"
    JUDGE_TEMPLATE = "judge_template"
    RUBRIC = "rubric"
    DATASET = "dataset"
    SANDBOX = "sandbox"


class Manifest(BaseModel):
    """Fingerprints of everything that shapes the evidence."""

    prompt: str
    system_model: str
    provider_sdk: str
    retriever: str
    corpus: str
    tools: str
    judge_model: str
    judge_template: str
    rubric: str
    dataset: str
    sandbox: str


def detect(before: Manifest, after: Manifest) -> set[Change]:
    a, b = before.model_dump(), after.model_dump()
    return {Change(k) for k in a if a[k] != b[k]}


class Impact(BaseModel):
    suites: set[str]
    rerecord_outputs: bool      # system responses must be new
    rerecord_verdicts: bool     # judge verdicts must be new
    revalidate_judges: bool


OUTPUT_CHANGES = {Change.PROMPT, Change.SYSTEM_MODEL,
                  Change.PROVIDER_SDK, Change.RETRIEVER,
                  Change.CORPUS, Change.TOOLS, Change.SANDBOX}
JUDGE_CHANGES = {Change.JUDGE_MODEL, Change.JUDGE_TEMPLATE,
                 Change.RUBRIC}

SUITES: dict[Change, set[str]] = {
    Change.PROMPT: {"critical", "task", "safety"},
    Change.SYSTEM_MODEL: {"critical", "task", "rag", "agent",
                          "safety", "reliability"},
    Change.PROVIDER_SDK: {"smoke"},
    Change.RETRIEVER: {"rag", "critical"},
    Change.CORPUS: {"rag", "critical"},
    Change.TOOLS: {"agent", "safety"},
    Change.JUDGE_MODEL: {"judge-validation"},
    Change.JUDGE_TEMPLATE: {"judge-validation"},
    Change.RUBRIC: {"judge-validation", "human-labels"},
    Change.DATASET: {"critical"},
    Change.SANDBOX: {"agent"},
}


def impact(changes: set[Change]) -> Impact:
    suites: set[str] = {"regression"} if changes else set()
    for c in changes:
        suites |= SUITES[c]
    outputs = bool(changes & OUTPUT_CHANGES)
    judges = bool(changes & JUDGE_CHANGES)
    return Impact(
        suites=suites,
        rerecord_outputs=outputs,
        rerecord_verdicts=outputs or judges,
        revalidate_judges=judges or bool(
            changes & {Change.SYSTEM_MODEL, Change.RETRIEVER,
                       Change.CORPUS}))


def regression_case(case_id: str, text: str,
                    objective_ids: list[str], failure_mode: str,
                    reference: Reference,
                    origin: Source) -> EvalCase:
    """A reviewed failure, promoted so it cannot recur quietly."""
    return EvalCase(
        case_id=case_id, input=text,
        objective_ids=objective_ids, reference=reference,
        source=origin, split=Split.DEV, reviewed=True,
        tags=["regression", f"fm:{failure_mode}"])
