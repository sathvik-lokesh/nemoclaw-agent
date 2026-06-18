"""
Live agent team — the same planner→researcher→writer→critic pipeline as the
scripted demo, but every worker is a real LLM call (Ollama or NIM), instrumented
into the trace (model, measured latency, output tokens). The plan is verified by
Fast Downward BEFORE any worker runs — the pre-execution gate in action on a live
run.

By default every role uses one local model (kept warm → fast on CPU). Override
per role to demonstrate cost-aware routing, e.g. planner/critic on NIM Nemotron
and workers on a local SLM.
"""

from __future__ import annotations

from src.agents.llm import LLM, make_llm
from src.agents.team import build_plan
from src.router import adaptive_route
from src.schema import AgentStep, Plan
from src.verifier.plan_verifier import verify


def _good_enough(text: str) -> bool:
    """Default writer acceptor (stand-in for the critic): a substantive answer."""
    return len(text.split()) >= 12 and text.strip().endswith((".", "!", "?"))


def default_roles(model: str = "ollama:qwen2.5:3b") -> dict[str, LLM]:
    return {r: make_llm(model) for r in ("planner", "researcher", "writer", "critic")}


class PlanBlocked(Exception):
    pass


def _step(sid, agent, role, reply, t0, **kw) -> tuple[AgentStep, float]:
    t1 = t0 + reply.latency_s
    note = f"tokens={reply.tokens}"
    if kw.get("note"):
        note = kw.pop("note") + f" tokens={reply.tokens}"
    step = AgentStep(
        step_id=sid, agent_id=agent, role=role, model=reply.model, status="ok",
        ts_start=round(t0, 2), ts_end=round(t1, 2), output=reply.text.strip(),
        note=note, **kw)
    return step, t1


def run_live(recorder, topic: str = "the safety benefits of highway guardrails",
             roles: dict[str, LLM] | None = None, num_predict: int = 120,
             writer_ladder: list[str] | None = None, llm_factory=make_llm,
             accept=None) -> Plan:
    """Run a live team. If `writer_ladder` is given, the writer subtask is routed
    adaptively (cheapest model first, escalating until `accept` passes), and every
    attempt is recorded — the downshift is visible in the trace."""
    roles = roles or default_roles()
    accept = accept or _good_enough
    plan = build_plan()
    recorder.record_plan(plan)

    # --- pre-execution gate: verify the plan before spending any tokens ---
    vr = verify(plan)
    if not vr.ok and vr.fd_available:
        raise PlanBlocked(vr.reason())

    t = 0.0
    sid = 0

    # planner (records the plan it intends; its NL rationale is the output)
    sid += 1
    r = roles["planner"].chat(
        [{"role": "user", "content":
          f"You are a planner. The task is to write a short brief on: {topic}. "
          f"Briefly state the 3-step plan (research, write, review)."}],
        num_predict=num_predict)
    s, t = _step(sid, "planner", "planner", r, t, note="emitted plan")
    recorder.record_step(s); planner_id = sid

    # researcher
    sid += 1
    r = roles["researcher"].chat(
        [{"role": "user", "content":
          f"You are a researcher. Give 3 concise factual bullet points about: {topic}."}],
        num_predict=num_predict)
    claims = ["research_done"] if r.text.strip() else []
    s, t = _step(sid, "researcher", "worker", r, t,
                 input_refs=[planner_id], claims_satisfied=claims)
    recorder.record_step(s); researcher_id = sid
    research_output = s.output

    # writer (consumes researcher output) — static or adaptively routed
    writer_msgs = [{"role": "user", "content":
        f"You are a writer. Using these research notes, write a 3-sentence brief "
        f"on {topic}.\n\nNotes:\n{research_output}"}]
    writer_id = researcher_id
    draft = ""

    if writer_ladder:
        replies: dict[str, object] = {}
        def _gen(spec):
            rep = llm_factory(spec).chat(writer_msgs, num_predict=num_predict)
            replies[spec] = rep
            return rep.text.strip(), {}
        res = adaptive_route(writer_ladder, _gen, accept)
        for att in res.attempts:
            sid += 1
            rep = replies[att.model]
            note = "router:accepted" if att.accepted else "router:rejected→escalate"
            s, t = _step(sid, "writer", "worker", rep, t, input_refs=[researcher_id],
                         claims_satisfied=["draft_written"] if att.accepted else [],
                         note=note)
            recorder.record_step(s)
            writer_id, draft = sid, rep.text.strip()
            if att.accepted:
                break
    else:
        sid += 1
        r = roles["writer"].chat(writer_msgs, num_predict=num_predict)
        s, t = _step(sid, "writer", "worker", r, t,
                     input_refs=[researcher_id], claims_satisfied=["draft_written"])
        recorder.record_step(s); writer_id = sid
        draft = s.output

    # critic (accept/reject)
    sid += 1
    r = roles["critic"].chat(
        [{"role": "user", "content":
          f"You are a critic. Reply ACCEPT if this brief is coherent and on-topic, "
          f"else REJECT with a reason.\n\nBrief:\n{draft}"}],
        num_predict=num_predict)
    accepted = "accept" in r.text.lower()[:40] or "lgtm" in r.text.lower()[:40]
    s, t = _step(sid, "critic", "critic", r, t, input_refs=[writer_id],
                 claims_satisfied=["draft_accepted"] if accepted else [])
    recorder.record_step(s)

    return plan
