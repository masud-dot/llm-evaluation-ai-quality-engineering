"""systems/compass/single_turn.py: single-turn Compass.

Every input that shapes the answer is part of the prompt text,
so it is also part of the replay key.
"""
from __future__ import annotations

from pathlib import Path

from aiqe.providers import Provider


class SingleTurnCompass:
    def __init__(self, provider: Provider, model: str,
                 system_prompt: str) -> None:
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt

    @classmethod
    def from_files(cls, provider: Provider, model: str,
                   prompt_path: Path) -> SingleTurnCompass:
        return cls(provider, model,
                   prompt_path.read_text(encoding="utf-8"))

    def render(self, question: str) -> str:
        return f"{self.system_prompt}\n\nCUSTOMER: {question}"

    def __call__(self, question: str) -> str:
        return self.provider.complete(
            self.model, self.render(question)).text
