"""
Adaptive model-downshift demo.

Shows the router trying the cheapest model first and escalating only when a
validator rejects the result — so frontier compute is spent only where needed.

  python3 -m src.route_demo            # offline, deterministic fakes (instant)
  python3 -m src.route_demo --live     # real Ollama generation, heuristic accept

The acceptor here is a stand-in for the critic agent; in a full run the critic's
ACCEPT/REJECT is the validator.
"""

from __future__ import annotations

import argparse

from src.router import adaptive_route

LADDER = ["ollama:qwen2.5:3b", "ollama:qwen2.5:7b", "nim:nvidia/nemotron-3-super-120b-a12b"]

try:
    from rich.console import Console
    _c = Console()
    def _say(m): _c.print(m)
except Exception:
    import re
    def _say(m): print(re.sub(r"\[/?[^\]]*\]", "", m))


def _fake_generate(task: str):
    """Smaller models give terse, lower-quality answers; bigger ones elaborate."""
    quality = {"ollama:qwen2.5:3b": "short", "ollama:qwen2.5:7b": "ok",
               "nim:nvidia/nemotron-3-super-120b-a12b": "great"}
    def gen(spec):
        q = quality.get(spec, "ok")
        text = {"short": "Guardrails help.",
                "ok": "Guardrails reduce run-off-road crash severity by redirecting vehicles.",
                "great": ("Guardrails redirect errant vehicles back toward the roadway, "
                          "absorbing energy to cut run-off-road fatalities; placement and "
                          "end-treatment design materially affect outcomes.")}[q]
        return text, {"tier": q}
    return gen


def _heuristic_accept(text: str) -> bool:
    """Stand-in critic: accept a substantive, multi-clause answer."""
    return len(text.split()) >= 12 and text.strip().endswith(".")


def main() -> None:
    ap = argparse.ArgumentParser(prog="nemoclaw-route")
    ap.add_argument("--live", action="store_true", help="use real Ollama generation")
    ap.add_argument("--task", default="Explain the safety benefit of highway guardrails.")
    args = ap.parse_args()

    if args.live:
        from src.agents.llm import make_llm
        ladder = ["ollama:qwen2.5:3b", "ollama:qwen2.5:7b"]
        def generate(spec):
            r = make_llm(spec).chat([{"role": "user", "content": args.task}], num_predict=120)
            return r.text.strip(), {"latency_s": round(r.latency_s, 1), "tokens": r.tokens}
    else:
        ladder = LADDER
        generate = _fake_generate(args.task)

    _say(f"[bold]adaptive downshift[/]  task: {args.task}")
    _say(f"[dim]ladder (cheap→expensive): {' → '.join(ladder)}[/]\n")

    res = adaptive_route(ladder, generate, _heuristic_accept)
    for a in res.attempts:
        mark = "[green]ACCEPT[/]" if a.accepted else "[red]reject[/]"
        meta = f" {a.meta}" if a.meta else ""
        _say(f"  {mark}  {a.model}{meta}")
    _say(f"\n[bold]{res.summary()}[/]")
    if res.succeeded:
        _say(f"[dim]shipped output:[/] {res.output}")


if __name__ == "__main__":
    main()
