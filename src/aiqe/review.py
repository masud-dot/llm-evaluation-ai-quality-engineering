"""aiqe/review.py: trace review and failure taxonomies."""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Trace(BaseModel):
    """One recorded request to the system under test."""

    trace_id: str
    case_id: str
    input: str
    output: str
    model: str
    prompt_version: str


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"


class Annotation(BaseModel):
    trace_id: str
    case_id: str
    reviewer: str
    verdict: Verdict
    note: str = ""
    failure_mode: str | None = None

    @model_validator(mode="after")
    def failures_need_a_note(self) -> Annotation:
        if self.verdict is Verdict.FAIL and not self.note.strip():
            raise ValueError(f"{self.trace_id}: fail needs a note")
        return self


def review_session(
    traces: Iterable[Trace],
    reviewer: str,
    ask: Callable[[str], str] = input,
) -> list[Annotation]:
    """Open coding: a verdict and a free-text note per trace."""
    out: list[Annotation] = []
    for t in traces:
        print(f"\n[{t.trace_id}] {t.input}\n---\n{t.output}")
        verdict = Verdict(ask("pass/fail? ").strip().lower())
        note = ""
        if verdict is Verdict.FAIL:
            note = ask("first thing that went wrong: ")
        out.append(Annotation(
            trace_id=t.trace_id, case_id=t.case_id,
            reviewer=reviewer, verdict=verdict, note=note))
    return out


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    STANDARD = "standard"


class FailureMode(BaseModel):
    id: str = Field(pattern=r"^FM-[A-Z]+(-[A-Z]+)*$")
    name: str
    definition: str
    include: list[str] = Field(min_length=1)
    exclude: list[str] = Field(default_factory=list)
    severity: Severity
    objective_ids: list[str] = Field(default_factory=list)


class Taxonomy(BaseModel):
    version: str
    modes: list[FailureMode]

    @model_validator(mode="after")
    def ids_are_unique(self) -> Taxonomy:
        ids = [m.id for m in self.modes]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate failure-mode ids")
        return self

    def gaps(self) -> list[str]:
        """Failure modes that no quality objective covers."""
        return [m.id for m in self.modes if not m.objective_ids]


UNCODED = "UNCODED"


def tally(annotations: Iterable[Annotation],
          taxonomy: Taxonomy) -> Counter[str]:
    known = {m.id for m in taxonomy.modes}
    counts: Counter[str] = Counter()
    for a in annotations:
        if a.verdict is Verdict.PASS:
            continue
        mode = a.failure_mode or UNCODED
        if mode != UNCODED and mode not in known:
            raise ValueError(f"{a.trace_id}: unknown {mode}")
        counts[mode] += 1
    return counts


WEIGHTS = {Severity.CRITICAL: 10, Severity.HIGH: 3,
           Severity.STANDARD: 1}


def prioritize(counts: Counter[str],
               taxonomy: Taxonomy) -> list[tuple[str, int, int]]:
    """(mode, count, priority), highest priority first."""
    sev = {m.id: m.severity for m in taxonomy.modes}
    rows = [(mode, n, n * WEIGHTS[sev[mode]])
            for mode, n in counts.items() if mode in sev]
    return sorted(rows, key=lambda r: (-r[2], r[0]))
