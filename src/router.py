"""
Adaptive model-downshift router — the active form of NVIDIA's SLM thesis.

Instead of statically assigning a model per agent, try the *cheapest* model on a
worker's subtask first and only escalate up a ladder if a validator (the critic)
rejects the result. The smallest model that passes is the one that ships — so
you pay frontier prices only for the subtasks that actually need them.

Backend-agnostic: `generate(spec) -> (text, meta)` and `accept(text) -> bool`
are injected, so the router is unit-testable with fakes and reusable for live
Ollama/NIM runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Attempt:
    model: str
    accepted: bool
    meta: dict = field(default_factory=dict)


@dataclass
class RoutingResult:
    model: str | None          # the model whose output was accepted (None if all failed)
    output: str
    attempts: list[Attempt]
    succeeded: bool

    @property
    def escalations(self) -> int:
        return len(self.attempts) - 1

    def summary(self) -> str:
        path = " → ".join(a.model for a in self.attempts)
        if self.succeeded:
            return (f"accepted by {self.model} after {self.escalations} escalation(s) "
                    f"[tried: {path}]")
        return f"all {len(self.attempts)} models rejected [tried: {path}]"


def adaptive_route(ladder: list[str],
                   generate: Callable[[str], tuple[str, dict]],
                   accept: Callable[[str], bool]) -> RoutingResult:
    """Try each model in `ladder` (cheap→expensive); stop at the first accepted."""
    if not ladder:
        raise ValueError("ladder must be non-empty")
    attempts: list[Attempt] = []
    output = ""
    for spec in ladder:
        output, meta = generate(spec)
        ok = accept(output)
        attempts.append(Attempt(model=spec, accepted=ok, meta=meta))
        if ok:
            return RoutingResult(spec, output, attempts, True)
    return RoutingResult(None, output, attempts, False)
