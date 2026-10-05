"""aiqe/datasets.py: evaluation cases, datasets, and cards."""
from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from aiqe.strategy import EvaluationPlan


class ReferenceKind(StrEnum):
    EXACT = "exact"          # one verifiable value
    ACCEPTABLE = "acceptable"  # any of a set of answers
    CRITERIA = "criteria"    # properties a good answer has
    LABEL = "label"          # a verdict assigned by a person


class Source(StrEnum):
    HANDWRITTEN = "handwritten"
    SYNTHETIC = "synthetic"
    PRODUCTION = "production"
    ADVERSARIAL = "adversarial"


class Split(StrEnum):
    DEV = "dev"
    HOLDOUT = "holdout"


class Reference(BaseModel):
    kind: ReferenceKind
    values: list[str] = Field(min_length=1)


class EvalCase(BaseModel):
    case_id: str = Field(pattern=r"^[a-z0-9-]+$")
    input: str
    objective_ids: list[str] = Field(min_length=1)
    reference: Reference | None = None
    source: Source
    split: Split = Split.DEV
    reviewed: bool = False
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def synthetic_needs_review_for_holdout(self) -> EvalCase:
        if (self.source is Source.SYNTHETIC
                and self.split is Split.HOLDOUT
                and not self.reviewed):
            raise ValueError(
                f"{self.case_id}: unreviewed synthetic case "
                "in holdout"
            )
        return self


class Dataset(BaseModel):
    name: str
    version: str
    cases: list[EvalCase]

    @model_validator(mode="after")
    def ids_are_unique(self) -> Dataset:
        ids = [c.case_id for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.name}: duplicate case ids")
        return self

    def content_hash(self) -> str:
        """Hash of case content, independent of file order."""
        h = hashlib.sha256()
        for case in sorted(self.cases, key=lambda c: c.case_id):
            blob = json.dumps(case.model_dump(mode="json"),
                              sort_keys=True,
                              separators=(",", ":"))
            h.update(blob.encode())
        return h.hexdigest()[:16]

    def check_objectives(self, plan: EvaluationPlan) -> None:
        known = {o.id for o in plan.objectives}
        for case in self.cases:
            unknown = set(case.objective_ids) - known
            if unknown:
                raise ValueError(
                    f"{case.case_id}: unknown objectives "
                    f"{sorted(unknown)}"
                )


def load_jsonl(path: Path, name: str, version: str) -> Dataset:
    lines = path.read_text(encoding="utf-8").splitlines()
    cases = [EvalCase.model_validate_json(x)
             for x in lines if x.strip()]
    return Dataset(name=name, version=version, cases=cases)


EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"\+?\d[\d \-]{8,}\d")


def pii_suspects(dataset: Dataset) -> list[str]:
    """Flag cases that look like they contain contact details.

    A tripwire for review, not a PII detector.
    """
    return [c.case_id for c in dataset.cases
            if EMAIL.search(c.input) or PHONE.search(c.input)]


class DatasetCard(BaseModel):
    name: str
    version: str
    content_hash: str
    purpose: str
    owner: str
    objectives: list[str]
    counts_by_source: dict[str, int]
    counts_by_split: dict[str, int]
    known_gaps: list[str]
    pii_handling: str


def build_card(ds: Dataset, purpose: str, owner: str,
               known_gaps: list[str],
               pii_handling: str) -> DatasetCard:
    by_source: dict[str, int] = {}
    by_split: dict[str, int] = {}
    objectives: set[str] = set()
    for c in ds.cases:
        by_source[c.source] = by_source.get(c.source, 0) + 1
        by_split[c.split] = by_split.get(c.split, 0) + 1
        objectives.update(c.objective_ids)
    return DatasetCard(
        name=ds.name, version=ds.version,
        content_hash=ds.content_hash(), purpose=purpose,
        owner=owner, objectives=sorted(objectives),
        counts_by_source=by_source, counts_by_split=by_split,
        known_gaps=known_gaps, pii_handling=pii_handling)
