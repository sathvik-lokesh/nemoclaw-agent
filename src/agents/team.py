"""
Demo agent team: planner -> researcher -> writer -> critic (research/write/review).

The MVP uses a *scripted* backend so the recorder, detectors and verifier are
fully reproducible offline and we can deliberately inject the failure modes the
detectors are meant to catch. The structure (each agent is a function that
emits an AgentStep) is the same one a live NIM/Ollama backend would slot into.

Scenarios:
  ok                 — everything succeeds, critic accepts, goal met.
  error_propagation  — researcher's tool fails (status=error) but the writer
                       still consumes its output and produces a draft anyway
                       (the silent failure the flight recorder exists to catch).
  dropped_handoff    — researcher never produces the `research_done` claim, so
                       the writer's precondition is never satisfied upstream.
  correlated         — researcher and writer run on the *same* model and both
                       fail the same subtask → correlated base-model risk.
"""

from __future__ import annotations

from src.schema import AgentStep, Plan, Subtask, ToolCall

SCENARIOS = ["ok", "error_propagation", "dropped_handoff", "correlated",
             "livelock", "conflicting", "contract"]

# scenarios with a bespoke step structure (handled by dedicated builders)
SPECIAL_SCENARIOS = {"livelock", "conflicting", "contract"}

# Model assignment per agent. `correlated` forces a shared model.
def _models(scenario: str) -> dict[str, str]:
    if scenario == "correlated":
        return {"planner": "nemotron-3-super", "researcher": "qwen2.5:3b",
                "writer": "qwen2.5:3b", "critic": "nemotron-3-super"}
    return {"planner": "nemotron-3-super", "researcher": "qwen2.5:3b",
            "writer": "llama3.2:3b", "critic": "nemotron-3-super"}


def build_plan() -> Plan:
    """The planner's intended plan for 'write a short brief on topic X'."""
    return Plan(
        goal=["draft_accepted"],
        initial=["topic_given"],
        subtasks=[
            Subtask(id="t_research", agent="researcher",
                    preconditions=["topic_given"], effects=["research_done"]),
            Subtask(id="t_write", agent="writer",
                    preconditions=["research_done"], effects=["draft_written"]),
            Subtask(id="t_review", agent="critic",
                    preconditions=["draft_written"], effects=["draft_accepted"]),
        ],
    )


def build_broken_plan() -> Plan:
    """A plan the verifier should REJECT before execution: the writer needs
    `research_done`, but the plan contains no subtask that produces it (the
    research step was dropped at planning time), so the goal is unreachable."""
    return Plan(
        goal=["draft_accepted"],
        initial=["topic_given"],
        subtasks=[
            # no t_research → research_done is never produced
            Subtask(id="t_write", agent="writer",
                    preconditions=["research_done"], effects=["draft_written"]),
            Subtask(id="t_review", agent="critic",
                    preconditions=["draft_written"], effects=["draft_accepted"]),
        ],
    )


def _run_livelock(recorder) -> Plan:
    """A worker loops: it emits the identical draft twice, never reaching review."""
    plan = build_plan()
    recorder.record_plan(plan)
    recorder.record_step(AgentStep(
        step_id=1, agent_id="planner", role="planner", model="nemotron-3-super",
        status="ok", ts_start=0.0, ts_end=1.0, output="plan: research -> write -> review"))
    recorder.record_step(AgentStep(
        step_id=2, agent_id="researcher", role="worker", model="qwen2.5:3b",
        status="ok", ts_start=1.0, ts_end=4.0, input_refs=[1],
        output="notes about topic X", claims_satisfied=["research_done"]))
    for i, (a, b) in enumerate([(4.0, 8.0), (8.0, 12.0)]):  # same output twice
        recorder.record_step(AgentStep(
            step_id=3 + i, agent_id="writer", role="worker", model="llama3.2:3b",
            status="ok", ts_start=a, ts_end=b, input_refs=[2],
            output="DRAFT v1 (unchanged)", claims_satisfied=[],
            note="rewrote but produced identical draft"))
    return plan


def _run_conflicting(recorder) -> Plan:
    """Two reviewers assert contradictory verdicts about the same draft."""
    plan = Plan(goal=["verdict=approved"], initial=["draft_written"], subtasks=[])
    recorder.record_plan(plan)
    recorder.record_step(AgentStep(
        step_id=1, agent_id="writer", role="worker", model="llama3.2:3b",
        status="ok", ts_start=0.0, ts_end=3.0, output="draft", claims_satisfied=["draft_written"]))
    recorder.record_step(AgentStep(
        step_id=2, agent_id="critic_a", role="critic", model="nemotron-3-super",
        status="ok", ts_start=3.0, ts_end=5.0, input_refs=[1],
        output="approve", claims_satisfied=["verdict=approved"]))
    recorder.record_step(AgentStep(
        step_id=3, agent_id="critic_b", role="critic", model="qwen2.5:7b",
        status="ok", ts_start=3.0, ts_end=5.2, input_refs=[1],
        output="reject", claims_satisfied=["verdict=rejected"],
        note="disagrees with critic_a"))
    return plan


def _run_contract(recorder) -> Plan:
    """A worker reports success while its own tool call failed."""
    plan = build_plan()
    recorder.record_plan(plan)
    recorder.record_step(AgentStep(
        step_id=1, agent_id="planner", role="planner", model="nemotron-3-super",
        status="ok", ts_start=0.0, ts_end=1.0, output="plan"))
    recorder.record_step(AgentStep(
        step_id=2, agent_id="researcher", role="worker", model="qwen2.5:3b",
        status="ok", ts_start=1.0, ts_end=4.0, input_refs=[1],
        tool_calls=[ToolCall(name="cite_check", args={}, ok=False,
                             error="citation API returned 404")],
        output="3 facts (citations unverified)", claims_satisfied=["research_done"],
        note="claimed research_done despite failed cite_check"))
    return plan


_SPECIAL_BUILDERS = {
    "livelock": _run_livelock,
    "conflicting": _run_conflicting,
    "contract": _run_contract,
}


def run_team(scenario: str, recorder) -> Plan:
    """Run the scripted team for `scenario`, recording every step. Returns the Plan."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
    if scenario in SPECIAL_SCENARIOS:
        return _SPECIAL_BUILDERS[scenario](recorder)

    models = _models(scenario)
    plan = build_plan()
    recorder.record_plan(plan)

    sid = 0

    # --- planner ---
    sid += 1
    recorder.record_step(AgentStep(
        step_id=sid, agent_id="planner", role="planner", model=models["planner"],
        status="ok", ts_start=0.0, ts_end=1.1, input_refs=[],
        output="plan: research -> write -> review",
        claims_satisfied=[], note="emitted plan",
    ))
    planner_id = sid

    # --- researcher ---
    sid += 1
    researcher_id = sid
    if scenario == "error_propagation":
        recorder.record_step(AgentStep(
            step_id=sid, agent_id="researcher", role="worker", model=models["researcher"],
            status="error", ts_start=1.1, ts_end=3.0, input_refs=[planner_id],
            tool_calls=[ToolCall(name="web_search", args={"q": "topic X"}, ok=False,
                                 error="HTTP 503 from search backend")],
            output="", claims_satisfied=[],
            note="search tool failed; no notes produced",
        ))
    elif scenario == "dropped_handoff":
        # research 'succeeds' procedurally but never asserts research_done
        recorder.record_step(AgentStep(
            step_id=sid, agent_id="researcher", role="worker", model=models["researcher"],
            status="ok", ts_start=1.1, ts_end=4.0, input_refs=[planner_id],
            tool_calls=[ToolCall(name="web_search", args={"q": "topic X"}, ok=True)],
            output="some unstructured notes",
            claims_satisfied=[],  # <-- research_done NOT asserted
            note="returned notes but never confirmed research_done",
        ))
    elif scenario == "correlated":
        recorder.record_step(AgentStep(
            step_id=sid, agent_id="researcher", role="worker", model=models["researcher"],
            status="error", ts_start=1.1, ts_end=3.5, input_refs=[planner_id],
            tool_calls=[ToolCall(name="json_extract", args={}, ok=False,
                                 error="model returned malformed JSON")],
            output="```\nnot json\n```", claims_satisfied=[],
            note="qwen2.5:3b failed JSON formatting",
        ))
    else:  # ok
        recorder.record_step(AgentStep(
            step_id=sid, agent_id="researcher", role="worker", model=models["researcher"],
            status="ok", ts_start=1.1, ts_end=4.0, input_refs=[planner_id],
            tool_calls=[ToolCall(name="web_search", args={"q": "topic X"}, ok=True)],
            output="3 sourced facts about topic X",
            claims_satisfied=["research_done"],
        ))

    # --- writer ---
    sid += 1
    writer_id = sid
    if scenario == "correlated":
        recorder.record_step(AgentStep(
            step_id=sid, agent_id="writer", role="worker", model=models["writer"],
            status="error", ts_start=3.5, ts_end=5.5, input_refs=[researcher_id],
            tool_calls=[ToolCall(name="json_extract", args={}, ok=False,
                                 error="model returned malformed JSON")],
            output="```\nnot json\n```", claims_satisfied=[],
            note="same model, same JSON failure as researcher",
        ))
    else:
        # In error_propagation the writer consumes a FAILED upstream step but
        # presses on — producing a draft from broken input (the silent failure).
        consumed_failed = scenario == "error_propagation"
        recorder.record_step(AgentStep(
            step_id=sid, agent_id="writer", role="worker", model=models["writer"],
            status="ok", ts_start=4.0, ts_end=7.2, input_refs=[researcher_id],
            output="draft brief on topic X",
            claims_satisfied=["draft_written"],
            note="consumed failed research and wrote anyway" if consumed_failed else "",
        ))

    # --- critic ---
    sid += 1
    if scenario in ("ok", "error_propagation"):
        accepted = scenario == "ok"
        recorder.record_step(AgentStep(
            step_id=sid, agent_id="critic", role="critic", model=models["critic"],
            status="ok", ts_start=7.2, ts_end=8.6, input_refs=[writer_id],
            output="LGTM" if accepted else "rejected: claims unsupported by sources",
            claims_satisfied=["draft_accepted"] if accepted else [],
            note="" if accepted else "draft cites no real research",
        ))
    # dropped_handoff / correlated: execution stalls before a useful review.

    return plan
