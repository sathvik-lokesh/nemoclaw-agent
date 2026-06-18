"""
Pre-execution plan verification gate.

  python3 -m src.verify_plan            # verifies the good demo plan -> PASS
  python3 -m src.verify_plan --broken   # verifies a dropped-research plan -> BLOCK

Exit code 0 = plan allowed, 1 = blocked.
"""

from __future__ import annotations

import argparse
import sys

from src.agents.team import build_broken_plan, build_plan
from src.verifier.plan_verifier import verify

try:
    from rich.console import Console
    _c = Console()
    def _say(msg): _c.print(msg)
except Exception:
    def _say(msg):
        import re
        print(re.sub(r"\[/?[^\]]*\]", "", msg))


def main() -> None:
    ap = argparse.ArgumentParser(prog="nemoclaw-verify")
    ap.add_argument("--broken", action="store_true", help="verify the broken demo plan")
    args = ap.parse_args()

    plan = build_broken_plan() if args.broken else build_plan()
    _say(f"[bold]verifying plan[/]  goal={plan.goal}  subtasks="
         f"{[s.id for s in plan.subtasks]}")
    res = verify(plan)

    if res.ok:
        _say(f"[bold green]✓ ALLOWED[/] — {res.reason()}")
        sys.exit(0)
    if not res.fd_available:
        _say(f"[yellow]⚠ {res.reason()}[/]")
        sys.exit(0)
    _say(f"[bold red]✗ BLOCKED before execution[/] — {res.reason()}")
    _say("[dim]no agent ran; no tokens spent.[/]")
    sys.exit(1)


if __name__ == "__main__":
    main()
