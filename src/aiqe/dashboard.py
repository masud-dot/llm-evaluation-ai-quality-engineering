"""aiqe/dashboard.py: a static quality status page."""
from __future__ import annotations

from html import escape

from aiqe.policy import GateResult
from aiqe.strategy import EvaluationPlan


def status_page(plan: EvaluationPlan, gate: GateResult,
                drift_alerts: list[str]) -> str:
    """One row per objective: met, not met, or no evidence."""
    failing = {f.objective: f.message for f in gate.findings}
    rows = []
    for o in plan.objectives:
        msg = failing.get(o.id)
        state = "met" if msg is None else (
            "no evidence" if msg.startswith("no ") else
            "not met")
        rows.append(
            f"<tr><td>{escape(o.id)}</td>"
            f"<td>{escape(o.risk)}</td>"
            f"<td>{escape(o.owner)}</td>"
            f"<td>{state}</td>"
            f"<td>{escape(msg or '')}</td></tr>")
    alerts = "".join(f"<li>{escape(a)}</li>"
                     for a in drift_alerts) or "<li>none</li>"
    return ("<html><body>"
            f"<h1>{escape(plan.system)} quality: "
            f"{escape(gate.verdict)}</h1>"
            f"<p>Plan v{escape(plan.version)}; "
            f"policy {escape(gate.policy)}</p>"
            "<table><tr><th>Objective</th><th>Risk</th>"
            "<th>Owner</th><th>Status</th><th>Finding</th></tr>"
            + "".join(rows) + "</table>"
            f"<h2>Drift alerts</h2><ul>{alerts}</ul>"
            "</body></html>")
