"""
Analyze a recorded trace and print the post-mortem.

  python3 -m src.analyze results/error_propagation.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.cost import summarize
from src.detectors.coordination import analyze
from src.recorder import Trace
from src.remediate import suggest_diversification
from src.report import render, render_cost, render_mitigation

DEFAULT_POOL = ["ollama:qwen2.5:3b", "ollama:llama3.2:3b", "ollama:phi3",
                "ollama:qwen2.5:7b", "nim:nvidia/nemotron-3-super-120b-a12b"]


def main() -> None:
    ap = argparse.ArgumentParser(prog="nemoclaw-analyze")
    ap.add_argument("trace", help="path to a .jsonl trace")
    ap.add_argument("--cost", action="store_true",
                    help="also show the SLM-routing cost summary")
    ap.add_argument("--mitigate", action="store_true",
                    help="suggest model diversification for correlated failures")
    args = ap.parse_args()

    trace = Trace.load(args.trace)
    findings = analyze(trace)
    render(trace, findings, scenario=Path(args.trace).stem)
    if args.mitigate:
        render_mitigation(suggest_diversification(trace, DEFAULT_POOL))
    if args.cost:
        render_cost(summarize(trace))


if __name__ == "__main__":
    main()
