"""
Core data model for nemoclaw-agent.

An execution is a Plan (what the planner agent intends) followed by an ordered
list of AgentSteps (what actually happened). Everything the detectors and the
formal verifier need is captured here — provenance (`input_refs`), the model
behind each step (for correlated-failure analysis), and the postconditions a
step claims to satisfy (for dropped-handoff / goal-reachability analysis).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ToolCall:
    name: str
    args: dict
    ok: bool = True
    error: str | None = None


@dataclass
class Subtask:
    """One node of the planner's proposed plan."""
    id: str
    agent: str                      # which agent role is expected to execute it
    preconditions: list[str] = field(default_factory=list)   # claims required before it can run
    effects: list[str] = field(default_factory=list)         # claims it produces if it succeeds


@dataclass
class Plan:
    """The planner agent's proposed plan, verified before execution."""
    goal: list[str]                 # claims that must hold for the task to be 'done'
    subtasks: list[Subtask] = field(default_factory=list)
    initial: list[str] = field(default_factory=list)         # claims true before anything runs

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "initial": self.initial,
            "subtasks": [asdict(s) for s in self.subtasks],
        }

    @staticmethod
    def from_dict(d: dict) -> "Plan":
        return Plan(
            goal=d["goal"],
            initial=d.get("initial", []),
            subtasks=[Subtask(**s) for s in d.get("subtasks", [])],
        )


@dataclass
class AgentStep:
    """One recorded step of execution — the unit the flight recorder writes."""
    step_id: int
    agent_id: str
    role: str                       # planner | worker | critic
    model: str                      # e.g. qwen2.5:3b, nemotron-3-super, scripted
    status: str = "ok"              # ok | error | timeout
    ts_start: float = 0.0
    ts_end: float = 0.0
    input_refs: list[str] = field(default_factory=list)   # ids of steps this consumed
    tool_calls: list[ToolCall] = field(default_factory=list)
    output: str = ""
    claims_satisfied: list[str] = field(default_factory=list)  # postconditions asserted
    note: str = ""                  # optional human annotation

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "AgentStep":
        tcs = [ToolCall(**t) for t in d.get("tool_calls", [])]
        d = {**d, "tool_calls": tcs}
        return AgentStep(**d)


def now() -> float:
    return time.monotonic()
