"""
Console reporter — renders an AuditReport as a human-readable table.
"""
from __future__ import annotations

from ai_critic.core.types import AuditReport, Verdict

_ICONS = {Verdict.PASS: "✅", Verdict.WARNING: "⚠️ ", Verdict.FAIL: "❌"}


def print_report(report: AuditReport) -> None:
    """Print a compact audit report to stdout."""
    v = report.overall_verdict
    icon = _ICONS[v]

    print()
    print("═" * 62)
    print(f"  MODEL AUDIT REPORT")
    print(f"  Problem type : {report.problem_type.value}")
    print(f"  Overall score: {report.overall_score:.3f}  {icon} {v.value.upper()}")
    print("═" * 62)
    print(f"  {'Dimension':<22} {'Score':>6}  {'Verdict':<10}  Summary")
    print("─" * 62)

    for dim in sorted(report.dimensions.values(), key=lambda d: d.score):
        icon_d = _ICONS[dim.verdict]
        summary_short = dim.summary[:35] + "…" if len(dim.summary) > 35 else dim.summary
        print(f"  {dim.name:<22} {dim.score:>6.3f}  {icon_d} {dim.verdict.value:<8}  {summary_short}")

    if report.top_suggestions:
        print()
        print("  TOP SUGGESTIONS")
        for i, s in enumerate(report.top_suggestions[:7], 1):
            print(f"  {i}. {s}")

    print("═" * 62)
    print()
