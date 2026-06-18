"""
The flight recorder — append-only JSONL trace of every AgentStep, plus the Plan
header. A Trace can be replayed from disk for post-hoc failure analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.schema import AgentStep, Plan


class TraceRecorder:
    """Records a single execution to a JSONL file.

    Line 0 is a header: {"type": "plan", ...}.
    Subsequent lines are steps: {"type": "step", ...}.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._steps: list[AgentStep] = []
        self._plan: Plan | None = None
        self._fh = self.path.open("w")

    def record_plan(self, plan: Plan) -> None:
        self._plan = plan
        self._fh.write(json.dumps({"type": "plan", **plan.to_dict()}) + "\n")
        self._fh.flush()

    def record_step(self, step: AgentStep) -> None:
        self._steps.append(step)
        self._fh.write(json.dumps({"type": "step", **step.to_dict()}) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "TraceRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Trace:
    """An execution trace loaded from disk (or built in-memory) for analysis."""

    def __init__(self, plan: Plan | None, steps: list[AgentStep]):
        self.plan = plan
        self.steps = steps

    @staticmethod
    def load(path: str | Path) -> "Trace":
        plan: Plan | None = None
        steps: list[AgentStep] = []
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            kind = rec.pop("type", "step")
            if kind == "plan":
                plan = Plan.from_dict(rec)
            else:
                steps.append(AgentStep.from_dict(rec))
        return Trace(plan, steps)

    def step_by_id(self, step_id: int) -> AgentStep | None:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None
