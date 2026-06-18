"""
Thin Fast Downward runner — the shared classical-planning core.

Mirrors the working invocation from ../av-scenario-forge (astar(lmcut), parse
"Solution found"), but runs in a scratch dir so it leaves no sas_plan/output
files behind and returns the plan steps when one is found.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

FAST_DOWNWARD = str(Path("~/fast_downward/fast-downward.py").expanduser())


class PlanResult:
    def __init__(self, solved: bool, plan: list[str], available: bool = True):
        self.solved = solved          # True if a plan was found
        self.plan = plan              # ordered action names
        self.available = available    # False if FD could not be invoked

    def __bool__(self) -> bool:
        return self.solved


def solve(domain_pddl: str, problem_pddl: str,
          fd_path: str = FAST_DOWNWARD, timeout: int = 30) -> PlanResult:
    """Run Fast Downward on the given domain/problem strings."""
    with tempfile.TemporaryDirectory(prefix="nemoclaw_fd_") as d:
        dpath = Path(d) / "domain.pddl"
        ppath = Path(d) / "problem.pddl"
        plan_f = Path(d) / "sas_plan"
        dpath.write_text(domain_pddl)
        ppath.write_text(problem_pddl)
        try:
            result = subprocess.run(
                ["python3", fd_path, "--plan-file", str(plan_f),
                 str(dpath), str(ppath), "--search", "astar(lmcut())"],
                capture_output=True, text=True, timeout=timeout, cwd=d,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return PlanResult(False, [], available=False)

        out = result.stdout + result.stderr
        if "Solution found" in out or plan_f.exists():
            plan = []
            if plan_f.exists():
                plan = [ln.strip("()").strip() for ln in plan_f.read_text().splitlines()
                        if ln.strip().startswith("(")]
            return PlanResult(True, plan)
        return PlanResult(False, [])
