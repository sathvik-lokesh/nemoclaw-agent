"""
Static HTML trace-replay dashboard — a single self-contained file (inline CSS,
no server, no JS deps). Renders the timeline, the root-cause banner, and an
optional cost panel for a recorded trace.

  python3 -m src.dashboard results/error_propagation.jsonl -o report.html
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from src.cost import summarize
from src.detectors.coordination import Finding, analyze
from src.recorder import Trace

_SEV_BG = {"critical": "#b00020", "high": "#c75300", "medium": "#9a7d00", "info": "#555"}


def _esc(x) -> str:
    return html.escape(str(x))


def _rows(trace: Trace) -> str:
    out = []
    for s in trace.steps:
        bad = s.status in ("error", "timeout")
        flag = "✗" if bad else ("·" if not s.output else "✓")
        cls = ' class="bad"' if bad else ""
        out.append(
            f"<tr{cls}><td>{s.step_id}</td><td>{_esc(s.agent_id)}</td>"
            f"<td>{_esc(s.role)}</td><td><code>{_esc(s.model)}</code></td>"
            f"<td>{_esc(s.status)} {flag}</td>"
            f"<td>{_esc(', '.join(map(str, s.input_refs)) or '—')}</td>"
            f"<td>{_esc(', '.join(s.claims_satisfied) or '—')}</td>"
            f"<td>{_esc((s.note or s.output)[:90])}</td></tr>")
    return "\n".join(out)


def _findings(findings: list[Finding]) -> str:
    if not findings:
        return '<div class="ok">✓ no coordination failures detected</div>'
    head = findings[0]
    parts = [f'<div class="root" style="background:{_SEV_BG.get(head.severity, "#555")}">'
             f'<b>ROOT CAUSE — {_esc(head.category)}</b><br>{_esc(head.message)}</div>']
    if len(findings) > 1:
        items = "".join(f"<li><b>{_esc(f.category)}</b> "
                        f"<span class='sev'>[{_esc(f.severity)}]</span> — {_esc(f.message)}</li>"
                        for f in findings[1:])
        parts.append(f"<details open><summary>other findings</summary><ul>{items}</ul></details>")
    return "\n".join(parts)


def _cost(trace: Trace) -> str:
    s = summarize(trace)
    rows = "".join(f"<tr><td><code>{_esc(m)}</code></td><td>{pm['steps']}</td>"
                   f"<td>{pm['tokens']}</td><td>${pm['cost']:.4f}</td></tr>"
                   for m, pm in s["per_model"].items())
    return (f'<div class="cost"><h3>SLM routing</h3>'
            f'<p><b>${s["actual_usd"]:.4f}</b> routed vs '
            f'<b>${s["all_frontier_usd"]:.4f}</b> all-frontier → '
            f'<b>{s["savings_pct"]:.0f}% cheaper</b> '
            f'({s["total_tokens"]} tok, {s["total_latency_s"]:.1f}s)</p>'
            f'<table><tr><th>model</th><th>steps</th><th>tokens</th><th>cost</th></tr>'
            f'{rows}</table></div>')


def render_html(trace: Trace, findings: list[Finding], title: str) -> str:
    goal = ", ".join(trace.plan.goal) if trace.plan else "—"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>nemoclaw — {_esc(title)}</title><style>
body{{font:14px/1.5 system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}}
h1{{font-size:1.4rem}} code{{background:#f0f0f0;padding:1px 4px;border-radius:3px}}
table{{border-collapse:collapse;width:100%;margin:.5rem 0}}
th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:#fafafa}} tr.bad{{background:#fff0f0}}
.root{{color:#fff;padding:12px 14px;border-radius:6px;margin:1rem 0}}
.ok{{color:#1b7f3b;font-weight:600;margin:1rem 0}}
.sev{{color:#888}} .cost{{margin-top:1.5rem;border-top:1px solid #eee;padding-top:.5rem}}
.meta{{color:#666}}
</style></head><body>
<h1>nemoclaw-agent — post-mortem: {_esc(title)}</h1>
<p class="meta">goal: <code>{_esc(goal)}</code> · {len(trace.steps)} steps</p>
{_findings(findings)}
<h3>Execution timeline</h3>
<table><tr><th>#</th><th>agent</th><th>role</th><th>model</th><th>status</th>
<th>inputs</th><th>claims</th><th>note / output</th></tr>
{_rows(trace)}
</table>
{_cost(trace)}
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(prog="nemoclaw-dashboard")
    ap.add_argument("trace", help="path to a .jsonl trace")
    ap.add_argument("-o", "--out", default=None, help="output .html path")
    args = ap.parse_args()

    trace = Trace.load(args.trace)
    findings = analyze(trace)
    title = Path(args.trace).stem
    out = Path(args.out) if args.out else Path(args.trace).with_suffix(".html")
    out.write_text(render_html(trace, findings, title))
    print(f"[nemoclaw] dashboard written -> {out}")


if __name__ == "__main__":
    main()
