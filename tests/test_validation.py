import pytest
from pydantic import ValidationError

from aiqe.annotation import output_hash
from aiqe.datasets import Split
from aiqe.evaluators import Score
from aiqe.evaluators.judge import JudgeConfig
from aiqe.runner import RunResult
from aiqe.validation import (Confusion, JudgeRecord, JudgeStatus,
                             Pair, Validation,
                             corrected_pass_rate, pair_verdicts,
                             split_labels)

CRIT = "no-guarantee"


def test_pair_verdicts_skips_stale_labels() -> None:
    run = RunResult(dataset="d", dataset_version="1",
                    dataset_hash="h", split=Split.HOLDOUT,
                    system="s",
                    outputs={"a": "new text", "b": "same"},
                    scores=[Score(case_id="a", evaluator="j",
                                  passed=True),
                            Score(case_id="b", evaluator="j",
                                  passed=False)])
    final = {("a", output_hash("old text"), CRIT): 1,
             ("b", output_hash("same"), CRIT): 0}
    pairs, stale = pair_verdicts(run, "j", final, CRIT)
    assert stale == 1
    assert pairs == [Pair(case_id="b", judge=False,
                          human=False)]


def test_confusion_rates() -> None:
    pairs = ([Pair(case_id="x", judge=True, human=True)] * 57
             + [Pair(case_id="x", judge=False, human=True)] * 3
             + [Pair(case_id="x", judge=True, human=False)] * 12
             + [Pair(case_id="x", judge=False, human=False)] * 18
             + [Pair(case_id="x", judge=None, human=False)] * 2)
    c = Confusion.from_pairs(pairs)
    assert (c.tp, c.fn, c.fp, c.tn, c.deferred) == (
        57, 3, 12, 18, 2)
    assert c.tpr == pytest.approx(0.95)
    assert c.tnr == pytest.approx(0.60)


def test_corrected_pass_rate() -> None:
    # true 0.80, tpr 0.95, tnr 0.60 -> observed 0.84
    observed = 0.80 * 0.95 + 0.20 * 0.40
    assert observed == pytest.approx(0.84)
    assert corrected_pass_rate(observed, 0.95, 0.60) == \
        pytest.approx(0.80)
    with pytest.raises(ValueError):
        corrected_pass_rate(0.9, 0.5, 0.5)


def test_split_is_stratified_and_deterministic() -> None:
    final = {(f"c{i}", "h", CRIT): int(i >= 10)
             for i in range(40)}
    dev, test = split_labels(final, 0.5, seed=7)
    assert sum(1 for v in test.values() if v == 0) == 5
    assert split_labels(final, 0.5, seed=7) == (dev, test)
    assert not set(dev) & set(test)


def cfg() -> JudgeConfig:
    return JudgeConfig(judge_id="j", model="m-2026-01-01",
                       template_path="t", template_sha="0" * 16,
                       rubric_id="r", rubric_version="1.0",
                       criterion_id=CRIT)


def test_record_requires_evidence_for_validated() -> None:
    with pytest.raises(ValidationError):
        JudgeRecord(config=cfg(), adapter="a",
                    sampling={"temperature": 0}, owner="o",
                    status=JudgeStatus.VALIDATED,
                    revalidate_on=[])
    conf = Confusion(tp=1, fn=0, fp=0, tn=1, deferred=0)
    with pytest.raises(ValidationError):
        JudgeRecord(config=cfg(), adapter="a", sampling={},
                    owner="o", status=JudgeStatus.VALIDATED,
                    validation=Validation(
                        label_set="l", rubric_version="2.0",
                        confusion=conf,
                        validated_on="2026-09-01"),
                    revalidate_on=[])
