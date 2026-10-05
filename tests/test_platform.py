from pathlib import Path

import pytest

from aiqe.dashboard import status_page
from aiqe.datasets import Dataset, load_jsonl
from aiqe.policy import Finding, GateResult, Severity, Verdict
from aiqe.registry import Registry
from aiqe.strategy import load_plan


def test_registry_is_append_only(tmp_path: Path) -> None:
    ds = load_jsonl(
        Path("datasets/compass/credit-claims-v1.jsonl"),
        "credit-claims", "1.0")
    reg = Registry(tmp_path / "registry.toml")
    first = reg.register(ds, "a.jsonl")
    assert reg.register(ds, "a.jsonl") == first
    changed = Dataset(name=ds.name, version="1.0",
                      cases=ds.cases[:-1])
    with pytest.raises(ValueError):
        reg.register(changed, "b.jsonl")
    assert reg.latest("credit-claims").content_hash == \
        ds.content_hash()


def test_status_page_escapes_and_reports() -> None:
    plan = load_plan(Path("plans/compass-plan.toml"))
    gate = GateResult(policy="compass-release v1.2",
                      verdict=Verdict.FAIL, findings=[
                          Finding(objective="OBJ-SAFE-02",
                                  severity=Severity.BLOCK,
                                  message="leak <b>x</b>"),
                          Finding(objective="OBJ-LAT-01",
                                  severity=Severity.BLOCK,
                                  message="no evidence for p95")])
    html = status_page(plan, gate, ["bulky_delivery growing"])
    assert "leak &lt;b&gt;x&lt;/b&gt;" in html
    assert "<td>no evidence</td>" in html
    assert html.count("<tr><td>OBJ-") == len(plan.objectives)
