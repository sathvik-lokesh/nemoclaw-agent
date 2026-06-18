"""
Framework adapter — make nemoclaw a *layer over any agent framework*.

Rather than depend on LangGraph/CrewAI directly, nemoclaw ingests a small
**normalized event stream** that any framework can emit from its callbacks. Once
ingested, the same detectors, verifier and cost model apply unchanged.

Normalized events (one dict each; a list, or JSONL on disk):
    {"kind": "plan", "goal": [...], "initial": [...], "subtasks": [...]}
    {"kind": "step", "agent": "writer", "role": "worker", "model": "...",
     "input_refs": [2], "output": "...", "status": "ok",
     "claims": ["draft_written"], "latency_s": 1.2, "tokens": 80,
     "tools": [{"name": "search", "args": {...}, "ok": true}]}

`FrameworkTracer` is a tiny callback-shaped helper: call `.on_step(...)` from a
framework's hook (e.g. LangGraph `on_chain_end` / CrewAI task callback) and it
accumulates normalized events you can hand to `ingest_events`.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.recorder import Trace
from src.schema import AgentStep, Plan, Subtask, ToolCall


def ingest_events(events: list[dict]) -> Trace:
    plan: Plan | None = None
    steps: list[AgentStep] = []
    sid = 0
    for ev in events:
        kind = ev.get("kind", "step")
        if kind == "plan":
            plan = Plan(
                goal=ev.get("goal", []),
                initial=ev.get("initial", []),
                subtasks=[Subtask(**s) for s in ev.get("subtasks", [])],
            )
        else:
            sid += 1
            lat = float(ev.get("latency_s", 0.0))
            note = ev.get("note", "")
            if "tokens" in ev:
                note = (note + f" tokens={int(ev['tokens'])}").strip()
            steps.append(AgentStep(
                step_id=ev.get("step_id", sid),
                agent_id=ev["agent"],
                role=ev.get("role", "worker"),
                model=ev.get("model", "unknown"),
                status=ev.get("status", "ok"),
                ts_start=float(ev.get("ts_start", 0.0)),
                ts_end=float(ev.get("ts_end", lat)),
                input_refs=ev.get("input_refs", []),
                tool_calls=[ToolCall(**t) for t in ev.get("tools", [])],
                output=ev.get("output", ""),
                claims_satisfied=ev.get("claims", []),
                note=note,
            ))
    return Trace(plan, steps)


def ingest_file(path: str | Path) -> Trace:
    events = [json.loads(line) for line in Path(path).read_text().splitlines()
              if line.strip()]
    return ingest_events(events)


class FrameworkTracer:
    """Accumulate normalized events from a framework's callbacks.

    Example (LangGraph-style)::

        tracer = FrameworkTracer()
        tracer.on_plan(goal=["answer"], initial=["question"], subtasks=[...])
        # inside a node / on_chain_end callback:
        tracer.on_step(agent="researcher", role="worker", model="qwen2.5:3b",
                       output=text, claims=["research_done"], tokens=83)
        trace = tracer.build()
    """

    def __init__(self) -> None:
        self.events: list[dict] = []
        self._t = 0.0

    def on_plan(self, **kw) -> None:
        self.events.append({"kind": "plan", **kw})

    def on_step(self, agent: str, role: str = "worker", model: str = "unknown",
                output: str = "", status: str = "ok", claims: list[str] | None = None,
                input_refs: list[int] | None = None, latency_s: float = 0.0,
                tokens: int = 0, tools: list[dict] | None = None) -> None:
        ev = {
            "kind": "step", "agent": agent, "role": role, "model": model,
            "output": output, "status": status, "claims": claims or [],
            "input_refs": input_refs or [], "ts_start": round(self._t, 2),
            "ts_end": round(self._t + latency_s, 2), "tokens": tokens,
            "tools": tools or [],
        }
        self._t += latency_s
        self.events.append(ev)

    def build(self) -> Trace:
        return ingest_events(self.events)
