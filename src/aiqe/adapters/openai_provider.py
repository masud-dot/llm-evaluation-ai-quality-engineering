"""OpenAI Responses API adapter (extra: providers)."""
from __future__ import annotations

import openai

from aiqe.providers import Completion


class OpenAIProvider:
    """temperature is fixed per instance; record it in the
    replay root (see replay_root) so a change is detected."""

    def __init__(self, client: openai.OpenAI,
                 temperature: float | None = None) -> None:
        self.client = client
        self.temperature = temperature

    def complete(self, model: str, prompt: str) -> Completion:
        r = self.client.responses.create(
            model=model, input=prompt,
            temperature=(openai.omit if self.temperature is None
                         else self.temperature))
        u = r.usage
        return Completion(
            text=r.output_text, model=r.model,
            input_tokens=u.input_tokens if u else 0,
            output_tokens=u.output_tokens if u else 0)
