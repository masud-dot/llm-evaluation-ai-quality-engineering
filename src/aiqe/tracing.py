"""aiqe/tracing.py: production traces and online sampling.

gen_ai.* names follow the OpenTelemetry GenAI semantic
conventions, which are in Development status: pin the version
you emit and review these names when upgrading.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable

from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer

from aiqe.providers import Completion, Provider

OPERATION = "gen_ai.operation.name"
PROVIDER = "gen_ai.provider.name"
REQUEST_MODEL = "gen_ai.request.model"
RESPONSE_MODEL = "gen_ai.response.model"
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"


class TracedProvider:
    """Wraps any Provider; one client span per model call."""

    def __init__(self, inner: Provider, tracer: Tracer,
                 provider_name: str) -> None:
        self.inner = inner
        self.tracer = tracer
        self.provider_name = provider_name

    def complete(self, model: str, prompt: str) -> Completion:
        with self.tracer.start_as_current_span(
                f"chat {model}", kind=SpanKind.CLIENT) as span:
            span.set_attribute(OPERATION, "chat")
            span.set_attribute(PROVIDER, self.provider_name)
            span.set_attribute(REQUEST_MODEL, model)
            try:
                out = self.inner.complete(model, prompt)
            except Exception as exc:
                span.set_attribute("error.type",
                                   type(exc).__name__)
                span.set_status(Status(StatusCode.ERROR))
                raise
            span.set_attribute(RESPONSE_MODEL, out.model)
            span.set_attribute(INPUT_TOKENS, out.input_tokens)
            span.set_attribute(OUTPUT_TOKENS, out.output_tokens)
            return out


def bucket(key: str) -> float:
    """Stable value in [0, 1) derived from an identifier."""
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def sample(conversation_id: str, tags: Iterable[str],
           base_rate: float,
           always: frozenset[str]) -> bool:
    """Deterministic sampling for online evaluation.

    Conversations carrying a high-risk tag are always sampled;
    the rest are sampled at base_rate by a stable hash, so the
    same conversation gets the same decision everywhere.
    """
    if always & set(tags):
        return True
    return bucket(conversation_id) < base_rate
