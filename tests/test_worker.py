import json
import sys
from pathlib import Path

sys.path.insert(0, "projects/p5-production-platform")
from worker import main  # noqa: E402


def test_worker_samples_scores_and_triages(tmp_path: Path) -> None:
    conv = [{"conversation_id": "c1", "customer_text": "Credit?",
             "answer": "You'll receive a credit.",
             "tags": ["credit"]},
            {"conversation_id": "c2", "customer_text": "Where?",
             "answer": "It is in transit.", "tags": []}]
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps(c) for c in conv))
    assert main(["--input", str(src),
                 "--scores", str(tmp_path / "s.jsonl"),
                 "--triage", str(tmp_path / "t.jsonl"),
                 "--base-rate", "0"]) == 0
    scores = (tmp_path / "s.jsonl").read_text().splitlines()
    triage = (tmp_path / "t.jsonl").read_text().splitlines()
    assert len(scores) == 2          # c1 only, two evaluators
    assert [json.loads(t)["conversation_id"] for t in triage] \
        == ["c1"]
