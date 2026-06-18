"""
nemoclaw-agent CLI.

  python3 -m src.run --scenario error_propagation
  python3 -m src.run --scenario ok --out results/ok.jsonl

Runs the demo team under a scenario and writes a JSONL trace. (Detectors and
the formal plan verifier read this trace — added next.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.agents.team import SCENARIOS, run_team
from src.recorder import TraceRecorder


def main() -> None:
    ap = argparse.ArgumentParser(prog="nemoclaw-agent")
    ap.add_argument("--backend", choices=["scripted", "live"], default="scripted")
    ap.add_argument("--scenario", choices=SCENARIOS, default="error_propagation",
                    help="(scripted backend) which failure mode to inject")
    ap.add_argument("--model", default="ollama:qwen2.5:3b",
                    help="(live backend) model spec for all roles, e.g. ollama:qwen2.5:3b")
    ap.add_argument("--topic", default="the safety benefits of highway guardrails",
                    help="(live backend) task topic")
    ap.add_argument("--adaptive", action="store_true",
                    help="(live backend) route the writer adaptively (downshift)")
    ap.add_argument("--writer-ladder", default="ollama:qwen2.5:3b,ollama:qwen2.5:7b",
                    help="(live --adaptive) comma-separated model ladder, cheap→expensive")
    ap.add_argument("--out", default=None, help="trace output path (.jsonl)")
    args = ap.parse_args()

    if args.backend == "live":
        from src.agents.live_team import PlanBlocked, default_roles, run_live
        out = Path(args.out) if args.out else Path("results") / "live.jsonl"
        ladder = args.writer_ladder.split(",") if args.adaptive else None
        with TraceRecorder(out) as rec:
            try:
                plan = run_live(rec, topic=args.topic, roles=default_roles(args.model),
                                writer_ladder=ladder)
            except PlanBlocked as e:
                print(f"[nemoclaw] ✗ plan BLOCKED before execution — {e}")
                return
        mode = f"adaptive[{args.writer_ladder}]" if args.adaptive else args.model
        print(f"[nemoclaw] backend=live model={mode}  goal={plan.goal}")
        print(f"[nemoclaw] trace written -> {out}")
        return

    out = Path(args.out) if args.out else Path("results") / f"{args.scenario}.jsonl"
    with TraceRecorder(out) as rec:
        plan = run_team(args.scenario, rec)

    print(f"[nemoclaw] backend=scripted scenario={args.scenario}  goal={plan.goal}")
    print(f"[nemoclaw] trace written -> {out}")


if __name__ == "__main__":
    main()
