"""Anthropic Messages API adapter (extra: providers).

In the frozen SDK (anthropic 1.6.0), messages.create takes no
sampling parameters, so this adapter sets none.
"""
from __future__ import annotations

import anthropic

from aiqe.providers import Completion


class AnthropicProvider:
    def __init__(self, client: anthropic.Anthropic,
                 max_tokens: int = 1024) -> None:
        self.client = client
        self.max_tokens = max_tokens

    def complete(self, model: str, prompt: str) -> Completion:
        m = self.client.messages.create(
            model=model, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in m.content
                       if b.type == "text")
        return Completion(
            text=text, model=m.model,
            input_tokens=m.usage.input_tokens,
            output_tokens=m.usage.output_tokens)
