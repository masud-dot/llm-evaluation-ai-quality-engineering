"""aiqe/validation.py: meta-evaluation of judges."""
from __future__ import annotations

import random
from enum import StrEnum

from pydantic import BaseModel, model_validator

from aiqe.annotation import Item, output_hash
from aiqe.evaluators.judge import JudgeConfig
from aiqe.runner import RunResult


class Pair(BaseModel):
    case_id: str
    judge: bool | None     # None: judge deferred
    human: bool


def pair_verdicts(run: RunResult, judge_name: str,
                  final: dict[Item, int],
                  criterion_id: str) -> tuple[list[Pair], int]:
    """Match judge scores to human labels on the same output.

    Returns the pairs and the number of labels skipped because
    the run's output differs from the output that was labelled.
    """
    judged = {s.case_id: s.passed for s in run.scores
              if s.evaluator == judge_name}
    pairs, stale = [], 0
    for (cid, h, crit), value in final.items():
        if crit != criterion_id or cid not in judged:
            continue
        if output_hash(run.outputs[cid]) != h:
            stale += 1
            continue
        pairs.append(Pair(case_id=cid, judge=judged[cid],
                          human=value == 1))
    return pairs, stale


class Confusion(BaseModel):
    """Judge verdicts against human verdicts; 'pass' positive."""

    tp: int  # both pass
    fn: int  # human pass, judge fail
    fp: int  # human fail, judge pass
    tn: int  # both fail
    deferred: int

    @classmethod
    def from_pairs(cls, pairs: list[Pair]) -> Confusion:
        c = dict(tp=0, fn=0, fp=0, tn=0, deferred=0)
        for p in pairs:
            if p.judge is None:
                c["deferred"] += 1
            elif p.human:
                c["tp" if p.judge else "fn"] += 1
            else:
                c["fp" if p.judge else "tn"] += 1
        return cls(**c)

    @property
    def tpr(self) -> float:
        """Share of human passes the judge also passes."""
        if self.tp + self.fn == 0:
            raise ValueError("no human passes to measure TPR on")
        return self.tp / (self.tp + self.fn)

    @property
    def tnr(self) -> float:
        """Share of human failures the judge also fails."""
        if self.tn + self.fp == 0:
            raise ValueError("no human failures to measure TNR on")
        return self.tn / (self.tn + self.fp)


def corrected_pass_rate(observed: float, tpr: float,
                        tnr: float) -> float:
    """Estimate the true pass rate from a judge's observed rate.

    observed = p * tpr + (1 - p) * (1 - tnr), solved for p.
    """
    informative = tpr + tnr - 1
    if informative <= 0:
        raise ValueError("judge is no better than chance")
    p = (observed + tnr - 1) / informative
    return min(1.0, max(0.0, p))


def split_labels(final: dict[Item, int], test_share: float,
                 seed: int) -> tuple[dict[Item, int],
                                     dict[Item, int]]:
    """Deterministic dev/test split, stratified by label."""
    rng = random.Random(seed)
    dev: dict[Item, int] = {}
    test: dict[Item, int] = {}
    for value in sorted(set(final.values())):
        keys = sorted(k for k, v in final.items() if v == value)
        rng.shuffle(keys)
        cut = round(len(keys) * test_share)
        test.update({k: value for k in keys[:cut]})
        dev.update({k: value for k in keys[cut:]})
    return dev, test


class JudgeStatus(StrEnum):
    UNVALIDATED = "unvalidated"
    VALIDATED = "validated"
    SUSPENDED = "suspended"


class Validation(BaseModel):
    label_set: str         # adjudicated labels, with version
    rubric_version: str
    confusion: Confusion
    validated_on: str      # ISO date


class JudgeRecord(BaseModel):
    """The judge configuration record (Table 11)."""

    config: JudgeConfig
    adapter: str
    sampling: dict[str, float | int | str]
    owner: str
    status: JudgeStatus = JudgeStatus.UNVALIDATED
    validation: Validation | None = None
    revalidate_on: list[str]

    @model_validator(mode="after")
    def validated_needs_evidence(self) -> JudgeRecord:
        v = self.validation
        if self.status is JudgeStatus.VALIDATED:
            if v is None:
                raise ValueError("validated judge needs evidence")
            if v.rubric_version != self.config.rubric_version:
                raise ValueError("validated on another rubric")
        return self
