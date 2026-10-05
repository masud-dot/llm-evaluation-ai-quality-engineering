import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter \
    import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from aiqe.providers import Completion
from aiqe.tracing import (INPUT_TOKENS, RESPONSE_MODEL,
                          TracedProvider, bucket, sample)


class Fake:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def complete(self, model: str, prompt: str) -> Completion:
        if self.fail:
            raise TimeoutError("slow")
        return Completion(text="ok", model=model + "-resolved",
                          input_tokens=12, output_tokens=3)


def setup() -> tuple[InMemorySpanExporter, TracerProvider]:
    exp = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exp))
    return exp, tp


def test_client_span_attributes() -> None:
    exp, tp = setup()
    p = TracedProvider(Fake(), tp.get_tracer("aiqe"), "replay")
    p.complete("m-2026", "hello")
    (span,) = exp.get_finished_spans()
    assert span.name == "chat m-2026"
    assert span.attributes is not None
    assert span.attributes[RESPONSE_MODEL] == "m-2026-resolved"
    assert span.attributes[INPUT_TOKENS] == 12
    assert span.end_time is not None and span.start_time


def test_error_is_recorded_and_raised() -> None:
    exp, tp = setup()
    p = TracedProvider(Fake(fail=True), tp.get_tracer("aiqe"),
                       "replay")
    with pytest.raises(TimeoutError):
        p.complete("m", "x")
    (span,) = exp.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes is not None
    assert span.attributes["error.type"] == "TimeoutError"


def test_sampling_is_stable_and_risk_aware() -> None:
    risky = frozenset({"credit", "escalated"})
    ids = [f"conv-{i}" for i in range(4000)]
    picked = [i for i in ids if sample(i, [], 0.05, risky)]
    assert 0.03 < len(picked) / len(ids) < 0.07
    assert picked == [i for i in ids
                      if sample(i, [], 0.05, risky)]
    assert sample("conv-1", ["credit"], 0.0, risky)
    assert 0 <= bucket("x") < 1
