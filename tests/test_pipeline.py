"""P4 integration: manifest -> plan -> suites -> gate.

All model behaviour here is a deterministic stub recorded
through RecordingProvider; nothing is a Compass measurement.
"""
import json
import shutil
import sys
import tomllib
from pathlib import Path

import pytest

from aiqe.ci import main as ci
from aiqe.datasets import (EvalCase, Reference, ReferenceKind,
                           Source, Split)
from aiqe.evaluators.judge import (Cascade, JudgeConfig,
                                   SingleJudge, load_template)
from aiqe.evaluators.code import CommitmentPreFilter
from aiqe.providers import Completion, RecordingProvider
from aiqe.rubrics import Rubric
from systems.compass.single_turn import SingleTurnCompass

sys.path.insert(0, "projects/p4-ci-quality-gates")
import run_suites as rs  # noqa: E402
from run_suites import main as suites  # noqa: E402

from aiqe.changes import Impact  # noqa: E402

REPO = Path.cwd()
CAREFUL = "A credit may apply once a claim is reviewed."
PROMISE = "Don't worry, you'll receive a credit."
DAMAGE = "Report damage within 14 days of delivery."


class StubModel:
    def complete(self, model: str, prompt: str) -> Completion:
        q = prompt.rsplit("CUSTOMER: ", 1)[1]
        promise = "PROMISE" in prompt
        text = DAMAGE if "damage" in q else (
            PROMISE if promise else CAREFUL)
        return Completion(text=text, model=model,
                          input_tokens=1, output_tokens=1)


class StubJudge:
    def complete(self, model: str, prompt: str) -> Completion:
        v = 0 if "will receive" in prompt else 1
        return Completion(
            text=json.dumps({"reasoning": "stub", "verdict": v}),
            model=model, input_tokens=1, output_tokens=1)


def cases() -> list[EvalCase]:
    out = [EvalCase(case_id=f"cr-{i:03d}",
                    input=f"Parcel {i} late. Credit?",
                    objective_ids=["OBJ-CREDIT-01"],
                    source=Source.HANDWRITTEN,
                    split=Split.HOLDOUT, reviewed=True,
                    tags=["credit"]) for i in range(400)]
    out += [EvalCase(case_id=f"dm-{i:02d}",
                     input=f"Item {i} arrived damaged, deadline?",
                     objective_ids=["OBJ-CLAIM-01"],
                     reference=Reference(kind=ReferenceKind.EXACT,
                                         values=["14 days"]),
                     source=Source.HANDWRITTEN,
                     split=Split.HOLDOUT, reviewed=True,
                     tags=["claim", "damage"]) for i in range(10)]
    out += [EvalCase(case_id="reg-001",
                     input="Priority sofa two days late. Credit?",
                     objective_ids=["OBJ-CREDIT-01"],
                     source=Source.PRODUCTION, split=Split.DEV,
                     reviewed=True,
                     tags=["regression", "fm:FM-CREDIT-PROMISE"])]
    return out


PROMPTS = {"neutral": "Credits need a reviewed claim.",
           "reworded": "Explain that credits follow review.",
           "degraded": "Reassure customers. PROMISE credits."}


def workspace(root: Path) -> None:
    for d in ["rubrics", "prompts", "gates"]:
        shutil.copytree(REPO / d, root / d)
    (root / "gates/overrides").mkdir()
    (root / "data").mkdir()
    (root / "data/cases.jsonl").write_text("\n".join(
        c.model_dump_json() for c in cases()) + "\n")
    tpl = "prompts/judges/credit-commitment-v1.txt"
    _, sha = load_template(root / tpl)
    (root / "judge.toml").write_text(
        f'judge_id = "credit-cascade-judge"\n'
        f'model = "stub-judge-2026-01-01"\n'
        f'template_path = "{tpl}"\ntemplate_sha = "{sha}"\n'
        'rubric_id = "credit-commitment"\n'
        'rubric_version = "1.0"\n'
        'criterion_id = "no-guarantee"\n')
    (root / "suites.toml").write_text(
        '[system]\nmodel = "stub-2026-01-01"\n'
        'prompt = "active-prompt.txt"\n'
        'replay = "rec/compass"\njudge_replay = "rec/judge"\n'
        'judge = "judge.toml"\n'
        'rubric = "rubrics/credit-commitment.toml"\n'
        'emails = []\nphones = []\nurls = []\n'
        + "".join(
            f'\n[[suites]]\nname = "{n}"\nkind = "{k}"\n'
            'dataset = "data/cases.jsonl"\n'
            'dataset_name = "synthetic"\nversion = "1.0"\n'
            f'split = "{sp}"\ntiers = ["pr"]\n'
            for n, k, sp in [("critical", "critical", "holdout"),
                             ("task", "critical", "holdout"),
                             ("safety", "critical", "holdout"),
                             ("regression", "regression",
                              "dev")]))
    lit = "".join(f'{k} = {{ value = "v1" }}\n' for k in [
        "system_model", "provider_sdk", "retriever", "corpus",
        "tools", "judge_model", "judge_template", "rubric",
        "dataset", "sandbox"])
    (root / "manifest.toml").write_text(
        'prompt = { files = ["active-prompt.txt"] }\n' + lit)
    # Record every prompt variant once, through the real classes.
    with (root / "rubrics/credit-commitment.toml").open("rb") as f:
        rubric = Rubric.model_validate(tomllib.load(f))
    with (root / "judge.toml").open("rb") as f:
        cfg = JudgeConfig.model_validate(tomllib.load(f))
    cfg = cfg.model_copy(update={
        "template_path": str(root / cfg.template_path)})
    for d in ["rec/compass", "rec/judge"]:
        (root / d).mkdir(parents=True)
    judge = Cascade("credit-cascade", CommitmentPreFilter(),
                    SingleJudge(RecordingProvider(
                        StubJudge(), root / "rec/judge"),
                        cfg, rubric))
    for text in PROMPTS.values():
        sys_ = SingleTurnCompass(RecordingProvider(
            StubModel(), root / "rec/compass"),
            "stub-2026-01-01", text)
        for c in cases():
            judge.evaluate(c, sys_(c.input))


def gate(root: Path, variant: str) -> str:
    (root / "active-prompt.txt").write_text(PROMPTS[variant])
    assert ci(["manifest", "--config", "manifest.toml",
               "--out", "build/manifest.json"]) == 0
    assert ci(["plan", "--base", "baselines/current/manifest.json",
               "--head", "build/manifest.json",
               "--out", "build/impact.json"]) == 0
    assert suites(["--impact", "build/impact.json", "--tier", "pr",
                   "--evidence", "build/evidence.json",
                   "--config", "suites.toml"]) == 0
    rc = ci(["gate", "--policy", "gates/compass-pr.toml",
             "--evidence", "build/evidence.json",
             "--overrides", "gates/overrides",
             "--report", "build/report.md"])
    return {0: "pass", 1: "fail", 2: "review"}[rc]


def test_degraded_change_blocked_neutral_passes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    # Baseline: measure the neutral prompt with everything routed.
    base = tmp_path / "baselines/current"
    base.mkdir(parents=True)
    (tmp_path / "active-prompt.txt").write_text(PROMPTS["neutral"])
    ci(["manifest", "--config", "manifest.toml",
        "--out", str(base / "manifest.json")])
    (base / "evidence.json").write_text(
        '{"values": {}, "paired": {}}')
    (tmp_path / "all.json").write_text(json.dumps({
        "suites": ["critical", "regression"],
        "rerecord_outputs": True, "rerecord_verdicts": True,
        "revalidate_judges": False}))
    rs.run_suites(tmp_path, tmp_path / "suites.toml",
                  Impact.model_validate_json(
                      (tmp_path / "all.json").read_text()),
                  "pr", base, tmp_path / "b/evidence.json")
    shutil.copy(tmp_path / "b/run-critical.json",
                base / "run-critical.json")
    shutil.copy(tmp_path / "b/evidence.json",
                base / "evidence.json")
    assert gate(tmp_path, "reworded") == "pass"
    assert gate(tmp_path, "degraded") == "fail"
    report = (tmp_path / "build/report.md").read_text()
    assert "REGRESSION-SET" in report
    assert "OBJ-CREDIT-01" in report
    # A missing recording fails closed instead of passing.
    PROMPTS["unrecorded"] = "Never recorded."
    with pytest.raises(LookupError):
        gate(tmp_path, "unrecorded")
