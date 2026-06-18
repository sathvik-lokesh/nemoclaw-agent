"""
Pre-execution plan verifier — the showpiece.

Compiles a planner agent's Plan into a STRIPS PDDL domain+problem and asks Fast
Downward whether the goal is reachable from the initial state given the
subtasks' precondition/effect structure. If no plan exists, execution is
*blocked before any agent runs*, and we name the unsatisfiable precondition.

Each claim becomes a 0-ary predicate (c_<claim>); each subtask becomes an action
whose precondition/effect are its claims. A plan existing ⇔ the goal is
achievable by some valid ordering of the subtasks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.schema import Plan
from src.verifier.fd_runner import solve


def _pred(claim: str) -> str:
    return "c_" + re.sub(r"[^a-zA-Z0-9_]", "_", claim)


def _all_claims(plan: Plan) -> set[str]:
    claims: set[str] = set(plan.initial) | set(plan.goal)
    for st in plan.subtasks:
        claims.update(st.preconditions)
        claims.update(st.effects)
    return claims


def compile_domain(plan: Plan) -> str:
    preds = "\n    ".join(f"({_pred(c)})" for c in sorted(_all_claims(plan)))
    actions = []
    for st in plan.subtasks:
        pre = " ".join(f"({_pred(c)})" for c in st.preconditions)
        eff = " ".join(f"({_pred(c)})" for c in st.effects)
        actions.append(f"""  (:action {st.id}
    :precondition (and {pre})
    :effect (and {eff}))""")
    return f"""(define (domain nemoclaw-coordination)
  (:requirements :strips)
  (:predicates
    {preds})
{chr(10).join(actions)}
)"""


def compile_problem(plan: Plan) -> str:
    init = "\n    ".join(f"({_pred(c)})" for c in sorted(set(plan.initial)))
    goal = " ".join(f"({_pred(c)})" for c in plan.goal)
    return f"""(define (problem nemoclaw-coord)
  (:domain nemoclaw-coordination)
  (:init
    {init})
  (:goal (and {goal})))"""


@dataclass
class VerifyResult:
    ok: bool                       # True if plan is executable (goal reachable)
    plan_steps: list[str]          # FD's ordering when ok
    unreachable_goals: list[str]   # goal claims that can't be achieved
    blocking_subtasks: list[str]   # subtasks whose preconditions can never hold
    fd_available: bool = True

    def reason(self) -> str:
        if self.ok:
            return "plan verified — goal reachable via: " + " → ".join(self.plan_steps)
        if not self.fd_available:
            return "Fast Downward unavailable — could not verify"
        bits = []
        if self.unreachable_goals:
            bits.append(f"goal {self.unreachable_goals} not achievable")
        if self.blocking_subtasks:
            bits.append(f"precondition never satisfiable for {self.blocking_subtasks}")
        return "; ".join(bits) or "no valid execution ordering exists"


def _reachable_claims(plan: Plan) -> set[str]:
    """Forward closure: which claims can ever become true (ignoring ordering)."""
    have = set(plan.initial)
    changed = True
    while changed:
        changed = False
        for st in plan.subtasks:
            if set(st.preconditions) <= have and not set(st.effects) <= have:
                have |= set(st.effects)
                changed = True
    return have


def verify(plan: Plan) -> VerifyResult:
    res = solve(compile_domain(plan), compile_problem(plan))
    if not res.available:
        return VerifyResult(False, [], [], [], fd_available=False)
    if res.solved:
        return VerifyResult(True, res.plan, [], [])

    # Not solvable — diagnose why, so the report can name the culprit.
    reachable = _reachable_claims(plan)
    unreachable_goals = [g for g in plan.goal if g not in reachable]
    blocking = [st.id for st in plan.subtasks
                if not set(st.preconditions) <= reachable]
    return VerifyResult(False, [], unreachable_goals, blocking)
