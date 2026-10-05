from pathlib import Path

import pytest
from pydantic import ValidationError

from aiqe.datasets import (Dataset, EvalCase, Source, Split,
                           build_card, load_jsonl, pii_suspects)
from aiqe.strategy import load_plan

DATA = Path("datasets/compass/credit-claims-v1.jsonl")


def test_load_and_link_to_plan() -> None:
    ds = load_jsonl(DATA, "credit-claims", "1.0")
    assert len(ds.cases) == 4
    ds.check_objectives(load_plan(Path("plans/compass-plan.toml")))


def test_hash_ignores_order_but_not_content() -> None:
    ds = load_jsonl(DATA, "credit-claims", "1.0")
    flipped = Dataset(name=ds.name, version=ds.version,
                      cases=list(reversed(ds.cases)))
    assert flipped.content_hash() == ds.content_hash()
    edited = ds.model_copy(deep=True)
    edited.cases[0].input += "?"
    assert edited.content_hash() != ds.content_hash()


def test_unknown_objective_rejected() -> None:
    ds = load_jsonl(DATA, "credit-claims", "1.0")
    ds.cases[0].objective_ids = ["OBJ-NOPE-01"]
    with pytest.raises(ValueError):
        ds.check_objectives(
            load_plan(Path("plans/compass-plan.toml")))


def test_unreviewed_synthetic_blocked_from_holdout() -> None:
    with pytest.raises(ValidationError):
        EvalCase(case_id="x-1", input="q",
                 objective_ids=["OBJ-CLAIM-01"],
                 source=Source.SYNTHETIC, split=Split.HOLDOUT)


def test_pii_tripwire_and_card() -> None:
    ds = load_jsonl(DATA, "credit-claims", "1.0")
    assert pii_suspects(ds) == []
    ds.cases[0].input += " Call me on +44 20 7946 0000"
    assert pii_suspects(ds) == ["cc-001"]
    card = build_card(ds, "p", "o", ["no ADR"], "none")
    assert card.counts_by_source == {"handwritten": 3,
                                     "synthetic": 1}
    assert card.counts_by_split == {"holdout": 3, "dev": 1}
