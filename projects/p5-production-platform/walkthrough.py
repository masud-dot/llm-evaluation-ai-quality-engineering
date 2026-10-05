"""One failure, from a production trace to a blocked release.

Every step uses the aiqe components built in Parts II to V.
The system under test is a stub standing in for Compass.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter \
    import InMemorySpanExporter
from pydantic import BaseModel

from aiqe.ci import resolve
from aiqe.datasets import (Dataset, Reference, ReferenceKind,
                           Split, load_jsonl)
from aiqe.evaluators.code import CommitmentPreFilter
from aiqe.evidence import EvidenceBuilder
from aiqe.online import evaluate_sampled
from aiqe.policy import GateResult, apply, load_policy
from aiqe.providers import Completion
from aiqe.registry import Registry
from aiqe.runner import run
from aiqe.stats import Paired
from aiqe.tracing import TracedProvider, sample
from aiqe.triage import Status, TriageItem, promote

PROMISE = "Don't worry, you'll receive a credit for the delay."
CAREFUL = "A credit may apply once a claim is reviewed."
RISKY = frozenset({"credit", "escalated"})


class StubCompass:
    """Stands in for Compass; says whatever it was built to."""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, model: str, prompt: str) -> Completion:
        return Completion(text=self.reply, model=model,
                          input_tokens=40, output_tokens=12)


class Carried(BaseModel):
    """Evidence carried from the last accepted gate run."""

    values: dict[str, float]
    paired: dict[str, Paired]


class Outcome(BaseModel):
    spans: int
    sampled: bool
    online_passed: bool | None
    item: Status
    case: str
    versions: list[str]
    blocked: str
    findings: list[str]
    fixed: str


def gate_for(system: StubCompass, ds: Dataset,
             carried: Carried,
             policy: Path) -> tuple[GateResult, str]:
    """Regression evidence is computed; everything else is
    carried from the last accepted gate run."""
    res = run(ds, Split.DEV, "candidate",
              lambda q: system.complete("m", q).text,
              [CommitmentPreFilter()])
    failed = sum(1 for s in res.scores if s.passed is False)
    ev = EvidenceBuilder()
    for k, v in carried.values.items():
        ev.value(k, v)
    for name, p in carried.paired.items():
        ev.comparison(name, p)
    ev.count("regression.failures", failed)
    result = apply(load_policy(policy), ev.build())
    outcome, _ = resolve(result, [], datetime.now().date())
    return result, outcome


def walkthrough(root: Path, carried: Carried) -> Outcome:
    now = datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc)
    # 1. Production: a traced conversation, sampled by risk tag.
    exporter = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    live = TracedProvider(StubCompass(PROMISE),
                          tp.get_tracer("compass"), "stub")
    question = "Sofa TW-5531 is two days late. Credit?"
    answer = live.complete("m-2026", question).text
    sampled = sample("conv-7731", ["credit"], 0.05, RISKY)
    # 2. Online evaluation flags an explicit commitment.
    scores = evaluate_sampled("conv-7731", question, answer,
                              ["OBJ-CREDIT-01"],
                              [CommitmentPreFilter()], now)
    # 3. Triage confirms it; 4. promotion de-identifies it.
    item = TriageItem(item_id="0107", conversation_id="conv-7731",
                      signal="online pre-filter failure",
                      failure_mode="FM-CREDIT-PROMISE",
                      objective_ids=["OBJ-CREDIT-01"])
    item = item.move(Status.CONFIRMED, "credit-lead",
                     "explicit promise on Priority delay")
    item, case = promote(
        item, "My sofa is two days late. Do I get a credit?",
        Reference(kind=ReferenceKind.CRITERIA,
                  values=["Promises no compensation"]))
    # 5. The dataset gains the case as a new registered version.
    base = load_jsonl(root / "datasets/compass/"
                      "credit-claims-v1.jsonl",
                      "credit-claims", "1.0")
    registry = Registry(root / "datasets/registry.toml")
    registry.register(base, "credit-claims-v1.jsonl")
    grown = Dataset(name=base.name, version="1.1",
                    cases=[*base.cases, case])
    registry.register(grown, "credit-claims-v1.1.jsonl")
    # 6. The unfixed candidate meets the gate; so does a fix.
    policy = root / "gates/compass-release.toml"
    blocked, outcome = gate_for(StubCompass(PROMISE), grown,
                                carried, policy)
    _, fixed = gate_for(StubCompass(CAREFUL), grown, carried,
                        policy)
    return Outcome(
        spans=len(exporter.get_finished_spans()),
        sampled=sampled, online_passed=scores[0].score.passed,
        item=item.status, case=case.case_id,
        versions=[e.version for e in registry.entries()],
        blocked=outcome,
        findings=[f.objective for f in blocked.findings],
        fixed=fixed)
