"""Draw the three measured figures from recorded evidence.

F9  judge agreement matrix      --confusion <json>
F10 variance across trials      --trials <json>
F16 multi-objective frontier    --candidates <json>

Each input must exist. Missing input is an error, never an
estimate: the book publishes no invented values.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from aiqe.cost import Candidate, Direction, pareto_front  # noqa: E402
from aiqe.stats import Trials, case_bootstrap  # noqa: E402
from aiqe.validation import Confusion  # noqa: E402

W, DARK, LIGHT = 4.75, "#1a1a1a", "#c8c8c8"
plt.rcParams.update({"font.family": "Lato", "font.size": 7.2})


def need(path: Path | None, what: str) -> Path:
    if path is None or not path.exists():
        raise SystemExit(f"missing {what}: record it first")
    return path


def f9(path: Path, out: Path) -> None:
    c = Confusion.model_validate_json(path.read_text())
    f, ax = plt.subplots(figsize=(W, 3.3)); ax.axis("off")
    cells = [("TP", c.tp, 0.3, 0.62), ("FN", c.fn, 0.72, 0.62),
             ("FP", c.fp, 0.3, 0.3), ("TN", c.tn, 0.72, 0.3)]
    for lab, n, x, y in cells:
        ax.add_patch(Rectangle((x - 0.17, y - 0.13), 0.34,
                               0.26, fc=LIGHT, ec=DARK))
        ax.text(x, y, f"{lab}\n{n}", ha="center", va="center")
    ax.text(0.51, 0.82, "Judge verdict: pass        fail",
            ha="center", fontweight="bold")
    ax.text(0.08, 0.46, "Human\nlabel", ha="center",
            va="center", fontweight="bold")
    ax.text(0.51, 0.1, f"TPR = {c.tpr:.3f}    TNR = {c.tnr:.3f}"
            f"    deferred = {c.deferred}", ha="center")
    f.savefig(out / "F09.png", dpi=300, bbox_inches="tight")
    plt.close(f)


def f10(path: Path, out: Path) -> None:
    trials: Trials = {k: list(v) for k, v in
                      json.loads(path.read_text()).items()}
    n = len(next(iter(trials.values())))
    rates = [sum(v[i] for v in trials.values()) / len(trials)
             for i in range(n)]
    iv = case_bootstrap(trials)
    flaky = sum(1 for v in trials.values() if len(set(v)) > 1)
    f, ax = plt.subplots(figsize=(W, 2.9))
    ax.axhspan(iv.low, iv.high, color=LIGHT,
               label="95% case-bootstrap interval")
    ax.plot(range(1, n + 1), rates, "o", color=DARK,
            label="pass rate per trial")
    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels([f"trial {i}" for i in range(n)])
    ax.set_ylabel("pass rate")
    ax.set_title(f"{len(trials)} cases, {flaky} flaky",
                 fontsize=7.5)
    ax.legend(fontsize=6.4, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    f.tight_layout(); f.savefig(out / "F10.png", dpi=300)
    plt.close(f)


def f16(path: Path, out: Path) -> None:
    rows = json.loads(path.read_text())
    cands = [Candidate(name=r["name"], metrics={
        "groundedness": r["groundedness"], "cost": r["cost"]})
        for r in rows]
    dirs: dict[str, Direction] = {"groundedness": "max",
                                 "cost": "min"}
    front = pareto_front(cands, dirs)
    f, ax = plt.subplots(figsize=(W, 3.1))
    for r in rows:
        on = r["name"] in front
        ok = r.get("pass3_lower", 1.0) >= r.get("pass3_limit",
                                                0.95)
        ax.plot(r["cost"], r["groundedness"],
                "o" if ok else "x",
                color=DARK if on else LIGHT, markersize=6)
        ax.annotate(r["name"], (r["cost"], r["groundedness"]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=6.4)
    pts = sorted(((r["cost"], r["groundedness"]) for r in rows
                  if r["name"] in front))
    ax.step([p[0] for p in pts], [p[1] for p in pts],
            where="post", color=DARK, lw=0.8)
    ax.set_xlabel("cost per task"); ax.set_ylabel("groundedness")
    ax.set_title("circles meet the reliability limit; "
                 "crosses do not", fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    f.tight_layout(); f.savefig(out / "F16.png", dpi=300)
    plt.close(f)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confusion", type=Path)
    ap.add_argument("--trials", type=Path)
    ap.add_argument("--candidates", type=Path)
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.all or a.confusion:
        f9(need(a.confusion, "judge validation confusion"), a.out)
    if a.all or a.trials:
        f10(need(a.trials, "repeated-trial verdicts"), a.out)
    if a.all or a.candidates:
        f16(need(a.candidates, "migration candidates"), a.out)
    print(f"figures written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
