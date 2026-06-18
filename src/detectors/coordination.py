"""
Coordination-failure detectors — post-hoc analysis over a recorded Trace.

Each detector returns Findings. A Finding names a category from the error
taxonomy, the culprit step(s), and a one-line root cause. The report layer
surfaces the single highest-severity Finding as the headline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.recorder import Trace
from src.schema import AgentStep

SEVERITY_ORDER = {"critical": 3, "high": 2, "medium": 1, "info": 0}

# How "root-cause-y" a category is. Specific mechanisms outrank the generic
# dropped-handoff consequence, which outranks the goal-unreached symptom.
CATEGORY_PRIORITY = {
    "error_propagation": 3, "contract_violation": 3, "conflicting_actions": 3,
    "livelock": 3, "correlated_failure": 3,
    "dropped_handoff": 1,
    "goal_unreached": 0,
}


@dataclass
class Finding:
    category: str
    severity: str            # critical | high | medium | info
    message: str
    steps: list[int] = field(default_factory=list)   # culprit step ids
    is_symptom: bool = False  # a consequence (e.g. goal unreached), not a root cause

    def rank(self) -> int:
        return SEVERITY_ORDER.get(self.severity, 0)

    def sort_key(self) -> tuple:
        # root causes first, then most-specific mechanism, then severity —
        # symptoms and generic consequences never headline over a real root cause
        return (not self.is_symptom, CATEGORY_PRIORITY.get(self.category, 2), self.rank())


def _failed(step: AgentStep) -> bool:
    return step.status in ("error", "timeout") or step.output.strip() == ""


def detect_error_propagation(trace: Trace) -> list[Finding]:
    """A step consumed (directly) the output of an upstream step that failed,
    yet did not itself fail — the failure was silently absorbed downstream."""
    findings: list[Finding] = []
    failed_ids = {s.step_id for s in trace.steps if _failed(s)}
    for s in trace.steps:
        if _failed(s):
            continue
        tainted = [r for r in s.input_refs if r in failed_ids]
        if tainted:
            up = ", ".join(f"step {r}" for r in tainted)
            findings.append(Finding(
                category="error_propagation", severity="critical",
                message=(f"{s.agent_id} (step {s.step_id}) consumed failed output from "
                         f"{up} but proceeded anyway — a silent failure that contaminates "
                         f"the final result."),
                steps=[*tainted, s.step_id],
            ))
    return findings


def detect_correlated_failure(trace: Trace) -> list[Finding]:
    """Two or more steps on the SAME model failed — base-model risk: agents
    sharing a backbone fail together, so retries/voting won't help."""
    by_model: dict[str, list[AgentStep]] = {}
    for s in trace.steps:
        if s.status in ("error", "timeout"):
            by_model.setdefault(s.model, []).append(s)
    findings: list[Finding] = []
    for model, steps in by_model.items():
        if len(steps) >= 2:
            ids = [s.step_id for s in steps]
            agents = ", ".join(sorted({s.agent_id for s in steps}))
            findings.append(Finding(
                category="correlated_failure", severity="high",
                message=(f"{len(steps)} agents on '{model}' failed together ({agents}). "
                         f"Correlated base-model risk — diversify the model for at least "
                         f"one of them."),
                steps=ids,
            ))
    return findings


def detect_dropped_handoff(trace: Trace) -> list[Finding]:
    """A subtask precondition is never produced by any executed step (and is not
    in the initial state) — an upstream handoff the plan assumed never happened."""
    if trace.plan is None:
        return []
    produced = set(trace.plan.initial)
    for s in trace.steps:
        produced.update(s.claims_satisfied)
    findings: list[Finding] = []
    for st in trace.plan.subtasks:
        missing = [p for p in st.preconditions if p not in produced]
        if missing:
            findings.append(Finding(
                category="dropped_handoff", severity="high",
                message=(f"subtask '{st.id}' ({st.agent}) needs {missing} but no executed "
                         f"step ever produced it — a dropped handoff."),
                steps=[],
            ))
    return findings


def detect_livelock(trace: Trace, step_budget_factor: int = 3) -> list[Finding]:
    """The team is stuck: an agent emits the same output more than once (a loop),
    or the run blows past a sane step budget without reaching the goal."""
    findings: list[Finding] = []

    # repeated (agent, output) → the agent is looping on the same result
    seen: dict[tuple[str, str], list[int]] = {}
    for s in trace.steps:
        out = s.output.strip()
        if not out:
            continue
        seen.setdefault((s.agent_id, out), []).append(s.step_id)
    for (agent, _out), ids in seen.items():
        if len(ids) >= 2:
            findings.append(Finding(
                category="livelock", severity="high",
                message=(f"{agent} produced an identical result {len(ids)}× "
                         f"(steps {ids}) — the team is looping without progress."),
                steps=ids))

    # step-budget blow-out relative to the plan size
    if trace.plan and trace.plan.subtasks:
        budget = len(trace.plan.subtasks) * step_budget_factor
        if len(trace.steps) > budget and not findings:
            findings.append(Finding(
                category="livelock", severity="high",
                message=(f"{len(trace.steps)} steps for a {len(trace.plan.subtasks)}-subtask "
                         f"plan (budget {budget}) — runaway loop / no termination."),
                steps=[s.step_id for s in trace.steps[budget:]]))
    return findings


def detect_conflicting_actions(trace: Trace) -> list[Finding]:
    """Two agents assert contradictory claims about shared state. Claims of the
    form `key=value` conflict when the same key is given different values."""
    values: dict[str, list[tuple[str, int]]] = {}   # key -> [(value, step_id)]
    for s in trace.steps:
        for claim in s.claims_satisfied:
            if "=" in claim:
                key, val = claim.split("=", 1)
                values.setdefault(key.strip(), []).append((val.strip(), s.step_id))
    findings: list[Finding] = []
    for key, pairs in values.items():
        distinct = {v for v, _ in pairs}
        if len(distinct) >= 2:
            ids = [sid for _, sid in pairs]
            findings.append(Finding(
                category="conflicting_actions", severity="high",
                message=(f"agents disagree on '{key}': {sorted(distinct)} asserted across "
                         f"steps {ids} — contradictory shared state."),
                steps=ids))
    return findings


def detect_contract_violation(trace: Trace) -> list[Finding]:
    """An agent reported success (status ok + claimed an effect) while one of its
    own tool calls failed — the step's self-report violates its tool contract."""
    findings: list[Finding] = []
    for s in trace.steps:
        if s.status in ("error", "timeout"):
            continue
        failed_tools = [t.name for t in s.tool_calls if not t.ok]
        if failed_tools and s.claims_satisfied:
            findings.append(Finding(
                category="contract_violation", severity="high",
                message=(f"{s.agent_id} (step {s.step_id}) claimed {s.claims_satisfied} "
                         f"but its tool(s) {failed_tools} failed — success reported on a "
                         f"broken contract."),
                steps=[s.step_id]))
    return findings


def detect_goal_unreached(trace: Trace) -> list[Finding]:
    """The plan's goal claims were never all satisfied by end of trace."""
    if trace.plan is None:
        return []
    produced = set(trace.plan.initial)
    for s in trace.steps:
        produced.update(s.claims_satisfied)
    missing = [g for g in trace.plan.goal if g not in produced]
    if missing:
        return [Finding(
            category="goal_unreached", severity="critical",
            message=f"execution ended without satisfying goal {missing}.",
            steps=[], is_symptom=True,
        )]
    return []


DETECTORS = [
    detect_error_propagation,
    detect_correlated_failure,
    detect_dropped_handoff,
    detect_livelock,
    detect_conflicting_actions,
    detect_contract_violation,
    detect_goal_unreached,
]


def analyze(trace: Trace) -> list[Finding]:
    findings: list[Finding] = []
    for det in DETECTORS:
        findings.extend(det(trace))
    findings.sort(key=lambda f: f.sort_key(), reverse=True)
    return findings
