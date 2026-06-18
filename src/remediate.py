"""
Correlated-failure mitigation.

The `correlated_failure` detector flags agents that failed together because they
share a base model — retries and majority-voting can't help, since the agents
make the *same* mistake. The fix is diversity: reassign at least one of the
colliding agents to a different model from a pool.

`suggest_diversification` reads a trace, finds models that ≥2 agents failed on,
and proposes a new per-agent model assignment that breaks the collision.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.recorder import Trace


@dataclass
class Reassignment:
    agent: str
    from_model: str
    to_model: str


def _bare(model: str) -> str:
    """Identity of a model regardless of backend-prefix: 'ollama:qwen2.5:3b',
    'qwen2.5:3b' and 'nim:nvidia/qwen2.5:3b' all share the bare tail."""
    m = model.split(":", 1)[1] if model.startswith(("ollama:", "nim:")) else model
    return m.split("/")[-1]


def _failing_groups(trace: Trace) -> dict[str, list[str]]:
    """model -> [agent_ids that failed on it], for models with ≥2 failing agents."""
    groups: dict[str, list[str]] = {}
    for s in trace.steps:
        if s.status in ("error", "timeout"):
            groups.setdefault(s.model, []).append(s.agent_id)
    return {m: a for m, a in groups.items() if len(set(a)) >= 2}


def suggest_diversification(trace: Trace, pool: list[str]) -> list[Reassignment]:
    """Keep one agent on the original model; move the rest onto distinct pool
    models not already in use, so the shared-model collision is broken."""
    in_use = {_bare(s.model) for s in trace.steps}
    reassignments: list[Reassignment] = []
    for model, agents in _failing_groups(trace).items():
        # alternatives = pool models that are genuinely different and not yet in use
        alternatives = [m for m in pool
                        if _bare(m) != _bare(model) and _bare(m) not in in_use]
        # first colliding agent keeps the model; reassign the others
        for agent in sorted(set(agents))[1:]:
            if alternatives:
                target = alternatives.pop(0)
                in_use.add(_bare(target))
            else:
                # pool exhausted — fall back to any genuinely different pool model
                target = next((m for m in pool if _bare(m) != _bare(model)), model)
            reassignments.append(Reassignment(agent, model, target))
    return reassignments
