"""Smoke tests: each scenario produces the expected headline finding, and the
plan verifier allows the good plan / blocks the broken one."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adapters.ingest import FrameworkTracer, ingest_events
from src.agents.team import build_broken_plan, build_plan, run_team
from src.cost import summarize, tier_of
from src.dashboard import render_html
from src.detectors.coordination import analyze
from src.recorder import Trace, TraceRecorder
from src.remediate import suggest_diversification
from src.router import adaptive_route
from src.verifier.plan_verifier import verify


def _trace_for(scenario, tmp_path):
    out = tmp_path / f"{scenario}.jsonl"
    with TraceRecorder(out) as rec:
        run_team(scenario, rec)
    return Trace.load(out)


def test_ok_has_no_findings(tmp_path):
    assert analyze(_trace_for("ok", tmp_path)) == []


def test_error_propagation_headline(tmp_path):
    findings = analyze(_trace_for("error_propagation", tmp_path))
    assert findings and findings[0].category == "error_propagation"


def test_correlated_headline(tmp_path):
    findings = analyze(_trace_for("correlated", tmp_path))
    assert findings[0].category == "correlated_failure"


def test_dropped_handoff_headline(tmp_path):
    findings = analyze(_trace_for("dropped_handoff", tmp_path))
    assert findings[0].category == "dropped_handoff"


def test_good_plan_allowed():
    assert verify(build_plan()).ok is True


def test_broken_plan_blocked():
    res = verify(build_broken_plan())
    assert res.ok is False
    assert "draft_accepted" in res.unreachable_goals


def test_tier_classification():
    assert tier_of("nvidia/nemotron-3-super-120b-a12b") == "frontier"
    assert tier_of("qwen2.5:3b") == "slm"


def test_slm_routing_is_cheaper(tmp_path):
    out = tmp_path / "ok.jsonl"
    with TraceRecorder(out) as rec:
        run_team("ok", rec)
    s = summarize(Trace.load(out))
    # scripted 'ok' run mixes frontier planner/critic with SLM workers
    assert s["all_frontier_usd"] >= s["actual_usd"]
    assert 0 <= s["savings_pct"] <= 100


# --- adaptive downshift router ---

def test_router_accepts_cheapest_passing():
    ladder = ["small", "big"]
    gen = lambda spec: ("good enough answer", {})
    res = adaptive_route(ladder, gen, accept=lambda t: True)
    assert res.succeeded and res.model == "small" and res.escalations == 0


def test_router_escalates_until_accepted():
    ladder = ["small", "mid", "big"]
    gen = lambda spec: (spec, {})
    res = adaptive_route(ladder, gen, accept=lambda t: t == "big")
    assert res.succeeded and res.model == "big" and res.escalations == 2


def test_router_all_reject():
    res = adaptive_route(["a", "b"], lambda s: (s, {}), accept=lambda t: False)
    assert res.succeeded is False and res.model is None


# --- correlated-failure mitigation ---

def test_mitigation_breaks_shared_model(tmp_path):
    out = tmp_path / "correlated.jsonl"
    with TraceRecorder(out) as rec:
        run_team("correlated", rec)
    reassigns = suggest_diversification(
        Trace.load(out), pool=["qwen2.5:3b", "llama3.2:3b", "phi3"])
    assert len(reassigns) == 1                      # one of the two colliding agents moved
    r = reassigns[0]
    assert r.from_model == "qwen2.5:3b" and r.to_model != "qwen2.5:3b"


# --- framework adapter ---

def test_ingest_events_builds_trace():
    tr = FrameworkTracer()
    tr.on_plan(goal=["done"], initial=["start"],
               subtasks=[{"id": "t1", "agent": "w", "preconditions": ["start"],
                          "effects": ["done"]}])
    tr.on_step(agent="w", role="worker", model="qwen2.5:3b",
               output="x", claims=["done"], latency_s=1.0, tokens=10)
    trace = tr.build()
    assert trace.plan.goal == ["done"]
    assert len(trace.steps) == 1 and trace.steps[0].claims_satisfied == ["done"]
    assert analyze(trace) == []                     # goal reached, no failures


def test_ingest_detects_failure_from_external_trace():
    events = [
        {"kind": "plan", "goal": ["g"], "initial": ["i"],
         "subtasks": [{"id": "t", "agent": "b", "preconditions": ["a"], "effects": ["g"]}]},
        {"kind": "step", "agent": "a", "role": "worker", "model": "m", "status": "error"},
        {"kind": "step", "agent": "b", "role": "worker", "model": "m",
         "input_refs": [1], "output": "ran anyway", "claims": ["g"]},
    ]
    findings = analyze(ingest_events(events))
    assert findings[0].category == "error_propagation"


# --- html dashboard ---

def test_dashboard_renders_html(tmp_path):
    out = tmp_path / "ep.jsonl"
    with TraceRecorder(out) as rec:
        run_team("error_propagation", rec)
    trace = Trace.load(out)
    page = render_html(trace, analyze(trace), "error_propagation")
    assert page.startswith("<!doctype html>")
    assert "ROOT CAUSE" in page and "writer" in page


# --- the three added detectors (taxonomy completeness) ---

@pytest.mark.parametrize("scenario,category", [
    ("livelock", "livelock"),
    ("conflicting", "conflicting_actions"),
    ("contract", "contract_violation"),
])
def test_new_detectors_headline(scenario, category, tmp_path):
    out = tmp_path / f"{scenario}.jsonl"
    with TraceRecorder(out) as rec:
        run_team(scenario, rec)
    findings = analyze(Trace.load(out))
    assert findings and findings[0].category == category


def test_all_seven_scenarios_run():
    from src.agents.team import SCENARIOS
    assert len(SCENARIOS) == 7


# --- adaptive routing wired into the live loop (fake LLM, no network) ---

class _FakeLLM:
    """Returns a terse answer for 'small' models, a substantive one for 'big'."""
    def __init__(self, spec):
        self.spec = spec
    def chat(self, messages, num_predict=120, temperature=0.2):
        from src.agents.llm import Reply
        if "7b" in self.spec or "nemotron" in self.spec:
            txt = ("Guardrails redirect errant vehicles back to the roadway and absorb "
                   "crash energy, cutting run-off-road fatalities substantially.")
        else:
            txt = "Guardrails help."
        return Reply(txt, 0.1, len(txt.split()), self.spec)


def test_live_writer_downshift_escalates(tmp_path):
    out = tmp_path / "live.jsonl"
    from src.agents.live_team import run_live
    with TraceRecorder(out) as rec:
        run_live(rec, roles={r: _FakeLLM("ollama:qwen2.5:3b")
                             for r in ("planner", "researcher", "writer", "critic")},
                 writer_ladder=["ollama:qwen2.5:3b", "ollama:qwen2.5:7b"],
                 llm_factory=_FakeLLM)
    trace = Trace.load(out)
    writer_steps = [s for s in trace.steps if s.agent_id == "writer"]
    # small model rejected then 7b accepted → two writer attempts recorded
    assert len(writer_steps) == 2
    assert "rejected" in writer_steps[0].note
    assert writer_steps[1].model.endswith("7b")
    assert "draft_written" in writer_steps[1].claims_satisfied
