"""aiqe/ci.py: the command-line gate used by CI."""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from datetime import date
from pathlib import Path

from pydantic import BaseModel

from aiqe.changes import Manifest, detect, impact
from aiqe.manifest import build_manifest
from aiqe.policy import (Evidence, GateResult, Severity, Verdict,
                         apply, load_policy)


class Override(BaseModel):
    """A recorded owner decision on a review finding."""

    objective: str
    approver: str
    rationale: str
    expires: date


def load_overrides(folder: Path) -> list[Override]:
    out = []
    for p in sorted(folder.glob("*.toml")):
        with p.open("rb") as fh:
            out.append(Override.model_validate(tomllib.load(fh)))
    return out


def resolve(result: GateResult, overrides: list[Override],
            today: date) -> tuple[str, list[Override]]:
    """Overrides may settle review findings, never blocks."""
    if result.verdict is Verdict.FAIL:
        return "fail", []
    if result.verdict is Verdict.PASS:
        return "pass", []
    live = {o.objective: o for o in overrides
            if o.expires >= today}
    needed = {f.objective for f in result.findings
              if f.severity is Severity.REVIEW}
    if needed <= set(live):
        return "pass-with-override", [live[o] for o in
                                      sorted(needed)]
    return "review", []


def report(result: GateResult, outcome: str,
           used: list[Override], ev: Evidence) -> str:
    lines = [f"## Quality gate: {outcome.upper()}", "",
             f"Policy: `{result.policy}`", ""]
    if result.findings:
        lines += ["| Objective | Severity | Finding |",
                  "|---|---|---|"]
        lines += [f"| {f.objective} | {f.severity} | "
                  f"{f.message} |" for f in result.findings]
        lines.append("")
    for o in used:
        lines.append(f"Override for {o.objective} by "
                     f"{o.approver} until {o.expires}: "
                     f"{o.rationale}")
    lines += ["", "| Evidence | Value |", "|---|---|"]
    lines += [f"| {k} | {v:.4g} |"
              for k, v in sorted(ev.values.items())]
    for name, p in sorted(ev.paired.items()):
        lines.append(f"| {name} (paired) | "
                     f"{p.difference.estimate:+.3f}, "
                     f"p={p.p_value:.3g}, n={p.n} |")
    return "\n".join(lines) + "\n"


EXIT = {"pass": 0, "pass-with-override": 0, "fail": 1,
        "review": 2}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aiqe.ci")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mf = sub.add_parser("manifest")
    mf.add_argument("--config", type=Path, required=True)
    mf.add_argument("--out", type=Path, required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--base", type=Path, required=True)
    pl.add_argument("--head", type=Path, required=True)
    pl.add_argument("--out", type=Path, required=True)
    gt = sub.add_parser("gate")
    gt.add_argument("--policy", type=Path, required=True)
    gt.add_argument("--evidence", type=Path, required=True)
    gt.add_argument("--overrides", type=Path, required=True)
    gt.add_argument("--report", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "manifest":
        m = build_manifest(Path.cwd(), a.config)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(m.model_dump_json(indent=1))
        return 0
    if a.cmd == "plan":
        base = Manifest.model_validate_json(a.base.read_text())
        head = Manifest.model_validate_json(a.head.read_text())
        imp = impact(detect(base, head))
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(imp.model_dump_json(indent=1))
        return 0
    ev = Evidence.model_validate_json(a.evidence.read_text())
    result = apply(load_policy(a.policy), ev)
    outcome, used = resolve(result, load_overrides(a.overrides),
                            date.today())
    a.report.write_text(report(result, outcome, used, ev))
    print(json.dumps({"outcome": outcome}))
    return EXIT[outcome]


if __name__ == "__main__":
    sys.exit(main())
