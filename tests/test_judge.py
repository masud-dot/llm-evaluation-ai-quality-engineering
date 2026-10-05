import tomllib
from pathlib import Path
from string import Template

import pytest

from aiqe.datasets import EvalCase, Source
from aiqe.evaluators.code import CommitmentPreFilter
from aiqe.evaluators.judge import (Cascade, JudgeConfig,
                                   PairwiseJudge, SingleJudge,
                                   load_template, parse_verdict)
from aiqe.providers import (Completion, RecordingProvider,
                            ReplayProvider)
from aiqe.rubrics import Rubric

TPL = Path("prompts/judges/credit-commitment-v1.txt")


class Scripted:
    """Stand-in provider returning scripted judge replies."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.prompts: list[str] = []

    def complete(self, model: str, prompt: str) -> Completion:
        self.prompts.append(prompt)
        return Completion(text=self.replies.pop(0), model=model,
                          input_tokens=1, output_tokens=1)


def rubric() -> Rubric:
    with Path("rubrics/credit-commitment.toml").open("rb") as f:
        return Rubric.model_validate(tomllib.load(f))


def config() -> JudgeConfig:
    _, sha = load_template(TPL)
    return JudgeConfig(judge_id="credit-judge-v1",
                       model="judge-snapshot-2026-01-01",
                       template_path=str(TPL), template_sha=sha,
                       rubric_id="credit-commitment",
                       rubric_version="1.0",
                       criterion_id="no-guarantee")


CASE = EvalCase(case_id="cc-001", input="Late parcel. Credit?",
                objective_ids=["OBJ-CREDIT-01"],
                source=Source.HANDWRITTEN)


def test_parse_verdict_strict() -> None:
    ok = '```json\n{"reasoning": "r", "verdict": 1}\n```'
    assert parse_verdict(ok) is not None
    assert parse_verdict('{"verdict": 2, "reasoning": "r"}') \
        is None
    assert parse_verdict("Verdict: pass") is None


def test_single_judge_scores_and_defers() -> None:
    p = Scripted(['{"reasoning": "promises", "verdict": 0}',
                  "I think it passes"])
    j = SingleJudge(p, config(), rubric())
    s = j.evaluate(CASE, "A credit is on its way.")
    assert s.passed is False and s.detail == "promises"
    assert "A credit is on its way." in p.prompts[0]
    assert "no-guarantee" in p.prompts[0]
    assert j.evaluate(CASE, "x").passed is None


def test_template_change_detected() -> None:
    bad = config().model_copy(update={"template_sha": "0" * 16})
    with pytest.raises(ValueError):
        SingleJudge(Scripted([]), bad, rubric())


def test_cascade_only_calls_judge_when_deferred() -> None:
    p = Scripted(['{"reasoning": "ok", "verdict": 1}'])
    c = Cascade("credit", CommitmentPreFilter(),
                SingleJudge(p, config(), rubric()))
    s1 = c.evaluate(CASE, "You'll receive a credit.")
    assert s1.passed is False and p.prompts == []
    s2 = c.evaluate(CASE, "A credit may apply after review.")
    assert s2.passed is True and len(p.prompts) == 1
    assert s2.evaluator == "credit"


def test_record_then_replay_judge(tmp_path: Path) -> None:
    live = Scripted(['{"reasoning": "r", "verdict": 1}'])
    rec = SingleJudge(RecordingProvider(live, tmp_path),
                      config(), rubric())
    first = rec.evaluate(CASE, "A credit may apply.")
    rep = SingleJudge(ReplayProvider(tmp_path), config(),
                      rubric())
    assert rep.evaluate(CASE, "A credit may apply.") == first


def test_pairwise_both_orders() -> None:
    t = Template("$input|$answer_a|$answer_b")
    pa = '{"winner": "A", "reasoning": "r"}'
    pb = '{"winner": "B", "reasoning": "r"}'
    consistent = PairwiseJudge(Scripted([pa, pb]), "m", t)
    assert consistent.compare(CASE, "x", "y") == "A"
    biased = PairwiseJudge(Scripted([pa, pa]), "m", t)
    assert biased.compare(CASE, "x", "y") == "inconsistent"
