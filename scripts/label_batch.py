"""Export a blind labelling batch, then import the labels.

    python scripts/label_batch.py export --run build/run.json
    python scripts/label_batch.py import --csv batch.csv \\
        --annotator "credit-lead"
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from aiqe.annotation import export_batch, import_labels
from aiqe.datasets import load_jsonl
from aiqe.rubrics import Rubric
from aiqe.runner import RunResult


def rubric_of(path: Path) -> Rubric:
    with path.open("rb") as fh:
        return Rubric.model_validate(tomllib.load(fh))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--run", type=Path, required=True)
    ex.add_argument("--dataset", type=Path,
                    default=Path("datasets/compass/"
                                 "credit-claims-v1.jsonl"))
    ex.add_argument("--name", default="credit-claims")
    ex.add_argument("--version", default="1.0")
    ex.add_argument("--rubric", type=Path,
                    default=Path("rubrics/"
                                 "credit-commitment.toml"))
    ex.add_argument("--out", type=Path,
                    default=Path("batch.csv"))
    ex.add_argument("--seed", type=int, default=0)
    im = sub.add_parser("import")
    im.add_argument("--csv", type=Path, required=True)
    im.add_argument("--annotator", required=True)
    im.add_argument("--rubric", type=Path,
                    default=Path("rubrics/"
                                 "credit-commitment.toml"))
    im.add_argument("--out", type=Path,
                    default=Path("labels.jsonl"))
    a = ap.parse_args(argv)
    rubric = rubric_of(a.rubric)
    if a.cmd == "export":
        run = RunResult.model_validate_json(a.run.read_text())
        ds = load_jsonl(a.dataset, a.name, a.version)
        export_batch(run, ds, rubric, a.out, a.seed)
        print(f"wrote {a.out}: fill in the value column only")
        return 0
    labels = import_labels(a.csv, a.annotator, rubric)
    a.out.write_text("".join(x.model_dump_json() + "\n"
                             for x in labels))
    print(f"{len(labels)} labels -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
