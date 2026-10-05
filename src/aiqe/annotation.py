"""aiqe/annotation.py: human labels, agreement, adjudication."""
from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import krippendorff
import numpy as np
from pydantic import BaseModel
from statsmodels.stats.inter_rater import cohens_kappa

from aiqe.datasets import Dataset
from aiqe.evaluators import Score
from aiqe.rubrics import Rubric, Scale
from aiqe.runner import RunResult


def output_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class HumanLabel(BaseModel):
    case_id: str
    output_hash: str   # the label applies to this exact output
    rubric_id: str
    rubric_version: str
    criterion_id: str
    annotator: str
    value: int
    note: str = ""


FIELDS = ["case_id", "output_hash", "input", "output",
          "criterion_id", "value", "note"]


def export_batch(run: RunResult, dataset: Dataset,
                 rubric: Rubric, path: Path,
                 seed: int = 0) -> None:
    """Write a blind, shuffled CSV for one annotator."""
    inputs = {c.case_id: c.input for c in dataset.cases}
    rows = [{"case_id": cid, "output_hash": output_hash(out),
             "input": inputs[cid], "output": out,
             "criterion_id": cr.id, "value": "", "note": ""}
            for cid, out in run.outputs.items()
            for cr in rubric.criteria]
    random.Random(seed).shuffle(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def import_labels(path: Path, annotator: str,
                  rubric: Rubric) -> list[HumanLabel]:
    allowed = {c.id: {lv.value for lv in c.levels}
               for c in rubric.criteria}
    labels = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row["value"].strip():
                continue    # not yet labelled
            v = int(row["value"])
            if v not in allowed[row["criterion_id"]]:
                raise ValueError(
                    f"{row['case_id']}: {v} is not a level")
            labels.append(HumanLabel(
                case_id=row["case_id"],
                output_hash=row["output_hash"],
                rubric_id=rubric.id,
                rubric_version=rubric.version,
                criterion_id=row["criterion_id"],
                annotator=annotator, value=v,
                note=row["note"]))
    return labels


Item = tuple[str, str, str]   # case, output hash, criterion


def by_item(labels: list[HumanLabel]
            ) -> dict[Item, dict[str, int]]:
    out: dict[Item, dict[str, int]] = defaultdict(dict)
    for lb in labels:
        key = (lb.case_id, lb.output_hash, lb.criterion_id)
        out[key][lb.annotator] = lb.value
    return dict(out)


def percent_agreement(labels: list[HumanLabel], a: str,
                      b: str) -> float:
    pairs = [(v[a], v[b]) for v in by_item(labels).values()
             if a in v and b in v]
    return sum(x == y for x, y in pairs) / len(pairs)


def kappa(labels: list[HumanLabel], a: str, b: str,
          n_levels: int) -> float:
    """Cohen's kappa for two annotators on shared items."""
    table = np.zeros((n_levels, n_levels))
    for v in by_item(labels).values():
        if a in v and b in v:
            table[v[a], v[b]] += 1
    return float(cohens_kappa(table).kappa)


def alpha(labels: list[HumanLabel], annotators: list[str],
          level: str = "nominal") -> float:
    """Krippendorff's alpha; tolerates missing labels."""
    items = list(by_item(labels).values())
    data = np.array([[v.get(a, np.nan) for v in items]
                     for a in annotators], dtype=float)
    return float(krippendorff.alpha(
        reliability_data=data, level_of_measurement=level))


class Adjudication(BaseModel):
    case_id: str
    output_hash: str
    criterion_id: str
    value: int
    adjudicator: str
    rationale: str


def final_labels(labels: list[HumanLabel],
                 adjudications: list[Adjudication]
                 ) -> dict[Item, int]:
    """Agreed values, or the adjudicated value on disagreement."""
    decided = {(a.case_id, a.output_hash, a.criterion_id):
               a.value for a in adjudications}
    out: dict[Item, int] = {}
    for key, votes in by_item(labels).items():
        values = set(votes.values())
        if len(values) == 1:
            out[key] = values.pop()
        elif key in decided:
            out[key] = decided[key]
        else:
            raise ValueError(f"unresolved disagreement: {key}")
    return out


def as_scores(final: dict[Item, int],
              rubric: Rubric) -> list[Score]:
    """Binary human verdicts as Scores, one per item."""
    if rubric.scale is not Scale.BINARY:
        raise ValueError("as_scores needs a binary rubric")
    return [Score(case_id=cid, evaluator=f"human:{rubric.id}",
                  passed=value == 1, detail=crit)
            for (cid, _h, crit), value in final.items()]
