"""Record model responses so evaluations can replay them.

Nothing here is needed to run the book's tests: they replay
fixtures. Use this when you want new recordings of your own.

    python scripts/record_fixtures.py --provider stub --dry-run
    python scripts/record_fixtures.py --provider openai \\
        --model <dated-snapshot> --trials 5
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from aiqe.datasets import Split, load_jsonl
from aiqe.evaluators.code import CommitmentPreFilter
from aiqe.evaluators.judge import (Cascade, JudgeConfig,
                                   SingleJudge)
from aiqe.providers import (Completion, Provider,
                            RecordingProvider, replay_root)
from aiqe.rubrics import Rubric
from systems.compass.single_turn import SingleTurnCompass

CAREFUL = ("A credit may apply once your claim has been "
           "reviewed. Damage claims must be made within 14 "
           "days of delivery.")


class StubProvider:
    """Deterministic stand-in for pipeline checks only."""

    def complete(self, model: str, prompt: str) -> Completion:
        return Completion(text=CAREFUL, model=model,
                          input_tokens=len(prompt) // 4,
                          output_tokens=len(CAREFUL) // 4)


class StubJudge:
    def complete(self, model: str, prompt: str) -> Completion:
        text = '{"reasoning": "stub", "verdict": 1}'
        return Completion(text=text, model=model,
                          input_tokens=len(prompt) // 4,
                          output_tokens=8)


def live(kind: str, temperature: float | None) -> Provider:
    if kind == "openai":
        import openai
        from aiqe.adapters.openai_provider import OpenAIProvider
        return OpenAIProvider(openai.OpenAI(), temperature)
    import anthropic
    from aiqe.adapters.anthropic_provider import AnthropicProvider
    return AnthropicProvider(anthropic.Anthropic())


def judge(root: Path, cfg_path: Path, rubric_path: Path,
          provider: Provider) -> Cascade:
    with cfg_path.open("rb") as fh:
        cfg = JudgeConfig.model_validate(tomllib.load(fh))
    with rubric_path.open("rb") as fh:
        rubric = Rubric.model_validate(tomllib.load(fh))
    return Cascade("credit-cascade", CommitmentPreFilter(),
                   SingleJudge(provider, cfg, rubric))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path,
                    default=Path("datasets/compass/"
                                 "credit-claims-v1.jsonl"))
    ap.add_argument("--name", default="credit-claims")
    ap.add_argument("--version", default="1.0")
    ap.add_argument("--split", choices=["dev", "holdout", "all"],
                    default="all")
    ap.add_argument("--provider", choices=["stub", "openai",
                                           "anthropic"],
                    required=True)
    ap.add_argument("--model", default="stub-model")
    ap.add_argument("--judge-model", default="stub-judge")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--prompt", type=Path,
                    default=Path("systems/compass/prompts/"
                                 "single-turn.txt"))
    ap.add_argument("--judge-config", type=Path,
                    default=Path("judges/credit-judge-v1.toml"))
    ap.add_argument("--rubric", type=Path,
                    default=Path("rubrics/"
                                 "credit-commitment.toml"))
    ap.add_argument("--out", type=Path,
                    default=Path("fixtures/replays/compass"))
    ap.add_argument("--judge-out", type=Path,
                    default=Path("fixtures/replays/judges"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    ds = load_jsonl(a.dataset, a.name, a.version)
    cases = [c for c in ds.cases if a.split == "all"
             or c.split is Split(a.split)]
    sampling = {} if a.temperature is None else {
        "temperature": a.temperature}
    base = replay_root(a.out, sampling)
    if a.dry_run:
        print(f"dataset {ds.name} {ds.version} "
              f"({ds.content_hash()}), {len(cases)} cases")
        print(f"system calls: {len(cases) * a.trials} "
              f"({a.trials} trial(s)) into {base}")
        print(f"judge calls: at most {len(cases) * a.trials} "
              f"into {a.judge_out}")
        print("no calls were made")
        return 0
    inner: Provider = (StubProvider() if a.provider == "stub"
                       else live(a.provider, a.temperature))
    jinner: Provider = (StubJudge() if a.provider == "stub"
                        else live(a.provider, a.temperature))
    for trial in range(a.trials):
        root = base if a.trials == 1 else base / f"trial-{trial}"
        root.mkdir(parents=True, exist_ok=True)
        a.judge_out.mkdir(parents=True, exist_ok=True)
        system = SingleTurnCompass.from_files(
            RecordingProvider(inner, root), a.model, a.prompt)
        cascade = judge(root, a.judge_config, a.rubric,
                        RecordingProvider(jinner, a.judge_out))
        for case in cases:
            cascade.evaluate(case, system(case.input))
        print(f"recorded trial {trial}: {len(cases)} cases "
              f"-> {root}")
    print("remember to record the model snapshot id in "
          "docs/VERSION_LOCK.md and judges/*.toml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
