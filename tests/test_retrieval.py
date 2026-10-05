from pathlib import Path

import pytest

from aiqe.datasets import Split
from aiqe.evaluators.judge import JudgeConfig, load_template
from aiqe.evaluators.retrieval import (
    Attribution, GroundednessJudge, Passage, RagCase, RagTrace,
    attribute, citation_validity, load_rag_jsonl, ndcg_at_k,
    precision_at_k, recall_at_k, reciprocal_rank)
from aiqe.providers import Completion

IDS = ["POL-CLM-2", "POL-CLM-1", "POL-FAQ-9"]


def test_rank_metrics() -> None:
    rel = {"POL-CLM-1"}
    assert recall_at_k(IDS, rel, 1) == 0.0
    assert recall_at_k(IDS, rel, 2) == 1.0
    assert precision_at_k(IDS, rel, 3) == pytest.approx(1 / 3)
    assert reciprocal_rank(IDS, rel) == 0.5
    assert ndcg_at_k(IDS, {"POL-CLM-1": 1.0}, 3) == \
        pytest.approx(1 / 1.5849625, rel=1e-6)
    perfect = ndcg_at_k(["POL-CLM-1"], {"POL-CLM-1": 2.0}, 3)
    assert perfect == 1.0


def test_attribution_quadrants() -> None:
    assert attribute(True, True) is Attribution.OK
    assert attribute(True, False) is Attribution.GENERATION
    assert attribute(False, False) is Attribution.RETRIEVAL
    assert attribute(False, True) is Attribution.UNGROUNDED


def trace(answer: str) -> RagTrace:
    return RagTrace(case_id="rag-001", query="q",
                    retrieved=[Passage(id=i, text="t")
                               for i in IDS], answer=answer)


def test_citation_validity() -> None:
    ok = trace("Within 14 days of delivery [POL-CLM-1].")
    assert citation_validity(ok).passed is True
    bad = trace("See [POL-CRD-1].")
    assert citation_validity(bad).passed is False
    assert citation_validity(trace("No cites.")).passed is None


def test_rag_dataset_keeps_relevant_ids() -> None:
    ds = load_rag_jsonl(
        Path("datasets/compass/policy-rag-v1.jsonl"),
        "policy-rag", "1.0")
    case = ds.cases[0]
    assert isinstance(case, RagCase)
    assert case.relevant_ids == ["POL-CLM-1"]
    assert "relevant_ids" in case.model_dump()
    edited = ds.model_copy(deep=True)
    c0 = edited.cases[0]
    assert isinstance(c0, RagCase)
    c0.relevant_ids = ["POL-CLM-2"]
    assert edited.content_hash() != ds.content_hash()
    assert all(c.split is Split.HOLDOUT for c in ds.cases)


class Scripted:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompt = ""

    def complete(self, model: str, prompt: str) -> Completion:
        self.prompt = prompt
        return Completion(text=self.reply, model=model,
                          input_tokens=1, output_tokens=1)


def test_groundedness_judge_sees_passages() -> None:
    tpl = Path("prompts/judges/groundedness-v1.txt")
    _, sha = load_template(tpl)
    cfg = JudgeConfig(judge_id="grounded-v1", model="m-2026",
                      template_path=str(tpl), template_sha=sha,
                      rubric_id="groundedness",
                      rubric_version="1.0", criterion_id="g")
    p = Scripted('{"reasoning": "not in passages", '
                 '"verdict": 0}')
    s = GroundednessJudge(p, cfg).evaluate_trace(
        trace("You have 30 days."))
    assert s.passed is False
    assert "[POL-CLM-1] t" in p.prompt
