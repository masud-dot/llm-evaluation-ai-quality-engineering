"""systems/compass/conversation.py: multi-turn Compass."""
from __future__ import annotations

from aiqe.conversation import Message, render
from aiqe.providers import Provider


class ConversationCompass:
    """The whole history is in the prompt, so it is also in
    the replay key (Chapter 11)."""

    def __init__(self, provider: Provider, model: str,
                 system_prompt: str) -> None:
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt

    def __call__(self, history: list[Message]) -> str:
        prompt = f"{self.system_prompt}\n\n{render(history)}"
        return self.provider.complete(self.model, prompt).text
