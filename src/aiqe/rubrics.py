"""aiqe/rubrics.py: rubrics with anchored levels."""
from __future__ import annotations

import textwrap
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Scale(StrEnum):
    BINARY = "binary"
    GRADED = "graded"
    PAIRWISE = "pairwise"


class Level(BaseModel):
    value: int
    anchor: str          # what an output at this level looks like
    example: str = ""    # a real output at this level


class Criterion(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    question: str        # asked of every output
    levels: list[Level] = Field(min_length=2)


class Rubric(BaseModel):
    id: str
    version: str
    objective_id: str
    scale: Scale
    criteria: list[Criterion] = Field(min_length=1)

    @model_validator(mode="after")
    def levels_match_scale(self) -> Rubric:
        for c in self.criteria:
            values = sorted(lv.value for lv in c.levels)
            if self.scale is Scale.BINARY and values != [0, 1]:
                raise ValueError(
                    f"{c.id}: binary needs levels 0 and 1")
            if values != list(range(len(values))):
                raise ValueError(
                    f"{c.id}: levels must be 0..n-1, each once")
        return self

    def render(self, width: int = 68) -> str:
        """Plain text for human reviewers and judge prompts."""
        lines = [f"Rubric {self.id} v{self.version} "
                 f"({self.objective_id}, {self.scale})"]
        for c in self.criteria:
            lines += textwrap.wrap(
                f"- {c.id}: {c.question}", width,
                subsequent_indent="  ")
            for lv in sorted(c.levels, key=lambda x: x.value):
                lines += textwrap.wrap(
                    f"{lv.value}: {lv.anchor}", width,
                    initial_indent="    ",
                    subsequent_indent="       ")
        return "\n".join(lines)
