"""
CLI post-mortem report. Uses `rich` if available, else degrades to plain text.
"""

from __future__ import annotations

from src.detectors.coordination import Finding
from src.recorder import Trace

try:
    from rich.console import Console
    from rich.table import Table
    _RICH = True
except Exception:                       # rich optional
    _RICH = False

_SEV_COLOR = {"critical": "bold red", "high": "red", "medium": "yellow", "info": "dim"}


def _timeline_rows(trace: Trace) -> list[tuple[str, ...]]:
    rows = []
    for s in trace.steps:
        flag = "✗" if s.status in ("error", "timeout") else ("·" if not s.output else "✓")
        rows.append((str(s.step_id), s.agent_id, s.model, s.status, flag,
                     (s.note or s.output)[:48]))
    return rows


def render(trace: Trace, findings: list[Finding], scenario: str | None = None) -> None:
    if _RICH:
        _render_rich(trace, findings, scenario)
    else:
        _render_plain(trace, findings, scenario)


def _render_rich(trace: Trace, findings, scenario) -> None:
    c = Console()
    title = f"nemoclaw-agent post-mortem" + (f" — {scenario}" if scenario else "")
    c.rule(f"[bold]{title}")

    t = Table(show_header=True, header_style="bold cyan")
    for col in ("#", "agent", "model", "status", "", "note / output"):
        t.add_column(col)
    for row in _timeline_rows(trace):
        style = "red" if row[3] in ("error", "timeout") else None
        t.add_row(*row, style=style)
    c.print(t)

    if not findings:
        c.print("[bold green]✓ no coordination failures detected[/]")
        return
    head = findings[0]
    c.print(f"\n[{_SEV_COLOR.get(head.severity)}]ROOT CAUSE[/]  "
            f"[{_SEV_COLOR.get(head.severity)}]{head.category}[/]: {head.message}")
    if len(findings) > 1:
        c.print("\n[dim]other findings:[/]")
        for f in findings[1:]:
            c.print(f"  [{_SEV_COLOR.get(f.severity)}]• {f.category}[/] — {f.message}")


def render_mitigation(reassignments) -> None:
    if not reassignments:
        return
    if _RICH:
        c = Console()
        c.print("\n[bold blue]MITIGATION[/]  break the shared-model collision:")
        for r in reassignments:
            c.print(f"  reassign [bold]{r.agent}[/]: "
                    f"[dim]{r.from_model}[/] → [green]{r.to_model}[/]")
    else:
        print("\nMITIGATION  break the shared-model collision:")
        for r in reassignments:
            print(f"  reassign {r.agent}: {r.from_model} -> {r.to_model}")


def render_cost(summary: dict) -> None:
    s = summary
    line = (f"cost: ${s['actual_usd']:.4f} routed  vs  ${s['all_frontier_usd']:.4f} "
            f"all-frontier  →  {s['savings_pct']:.0f}% cheaper   "
            f"({s['total_tokens']} tok, {s['total_latency_s']:.1f}s)")
    if _RICH:
        c = Console()
        c.print(f"\n[bold magenta]SLM ROUTING[/]  {line}")
        for m, pm in s["per_model"].items():
            c.print(f"  [dim]{m:<22}[/] {pm['steps']} steps  {pm['tokens']:>5} tok  "
                    f"${pm['cost']:.4f}")
    else:
        print(f"\nSLM ROUTING  {line}")
        for m, pm in s["per_model"].items():
            print(f"  {m:<22} {pm['steps']} steps  {pm['tokens']:>5} tok  ${pm['cost']:.4f}")


def _render_plain(trace: Trace, findings, scenario) -> None:
    title = "nemoclaw-agent post-mortem" + (f" — {scenario}" if scenario else "")
    print("=" * len(title)); print(title); print("=" * len(title))
    print(f"{'#':>2}  {'agent':<11}{'model':<18}{'status':<8}  note/output")
    for r in _timeline_rows(trace):
        print(f"{r[0]:>2}  {r[1]:<11}{r[2]:<18}{r[3]:<8}{r[4]} {r[5]}")
    print()
    if not findings:
        print("OK: no coordination failures detected"); return
    head = findings[0]
    print(f"ROOT CAUSE [{head.severity}] {head.category}: {head.message}")
    for f in findings[1:]:
        print(f"  - {f.category} [{f.severity}]: {f.message}")
