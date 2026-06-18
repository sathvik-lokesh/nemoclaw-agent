# nemoclaw-agent

> A **reliability layer for multi-agent systems** — a flight recorder that catches
> coordination failures, plus a formal plan verifier that blocks a provably-broken plan
> **before any agent runs**.

An open take on NVIDIA's **NemoClaw** (the orchestration + policy-enforcement layer of the
enterprise agent platform shipped at GTC 2026). Everyone is building *teams* of agents; far
fewer are addressing the problem the field openly calls underexplored — **multi-agent
reliability and correctness**: errors propagate between agents, agents on a shared base model
fail in correlated ways, and handoffs silently get dropped.

nemoclaw-agent sits *over* an agent team (a layer, not a framework), records what actually
happened, and tells you **which agent broke the chain and why**.

## What it does

| Stage | What it catches |
|-------|-----------------|
| **Pre-execution plan verifier** (Fast Downward) | A plan whose goal is unreachable / preconditions can never hold — blocked before a single token is spent |
| **Flight recorder** (JSONL trace) | Full provenance: per-step model, inputs consumed, claims asserted, tool calls |
| **Coordination-failure detector** | `error_propagation`, `correlated_failure`, `dropped_handoff`, `livelock`, `conflicting_actions`, `contract_violation` (+ `goal_unreached` symptom) |

The plan verifier reuses the neuro-symbolic pipeline from the sibling project
[`av-scenario-forge`](../av-scenario-forge): compile to PDDL → solve with **Fast Downward**.
That classical-planning core is what lets nemoclaw *prove* a plan is broken rather than
discover it at runtime — something pure-LLM agent stacks can't do.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # rich (core is stdlib); pytest + openai optional
make test                                # 20 tests  (or: pytest -q)

# 1) Verify a plan BEFORE running it (the showpiece)
python3 -m src.verify_plan               # good plan  -> ✓ ALLOWED (shows FD's ordering)
python3 -m src.verify_plan --broken      # broken plan -> ✗ BLOCKED, names the bad precondition

# 2a) Run an instrumented agent team (scripted backend — reproducible failure modes)
python3 -m src.run --scenario error_propagation

# 2b) ...or run REAL agents on local models (Ollama), gated by the plan verifier
python3 -m src.run --backend live --model ollama:qwen2.5:3b

# 3) Replay the trace and get the post-mortem (+ cost, + mitigation)
python3 -m src.analyze results/error_propagation.jsonl
python3 -m src.analyze results/live.jsonl --cost
python3 -m src.analyze results/correlated.jsonl --mitigate   # suggests model diversification

# 4) Adaptive model downshift — cheapest model that passes the critic
python3 -m src.route_demo                 # offline, deterministic
python3 -m src.run --backend live --adaptive   # downshift wired into a real run

# 5) Standalone HTML dashboard for a trace
python3 -m src.dashboard results/error_propagation.jsonl -o report.html
```

Example post-mortem (`error_propagation`):

```
ROOT CAUSE  error_propagation: writer (step 3) consumed failed output from step 2
but proceeded anyway — a silent failure that contaminates the final result.
```

## Demo scenarios

The MVP uses a **scripted backend** so failures are reproducible offline and the detectors are
testable without any API key or GPU. Each scenario injects one failure mode:

| Scenario | Injected fault | Detected as |
|----------|----------------|-------------|
| `ok` | none | *(clean)* |
| `error_propagation` | researcher's tool fails; writer presses on | `error_propagation` |
| `dropped_handoff` | researcher never asserts `research_done` | `dropped_handoff` |
| `correlated` | two agents on the same model fail together | `correlated_failure` |
| `livelock` | writer re-emits an identical draft, looping | `livelock` |
| `conflicting` | two reviewers assert opposite verdicts | `conflicting_actions` |
| `contract` | agent claims success while its tool call failed | `contract_violation` |

## Live backend & cost-aware routing

`--backend live` runs the team on real models — Ollama locally (`ollama:qwen2.5:3b`) or NVIDIA
NIM / Nemotron (`nim:nvidia/nemotron-3-super-120b-a12b`, needs `NIM_API_KEY`) — with each call
instrumented (model, measured latency, output tokens). `analyze --cost` then reports the run's
cost against an all-frontier counterfactual, illustrating NVIDIA's SLM thesis (workers on small
models cut agent cost sharply at comparable task success). Rates in `src/cost.py` are
illustrative; the *ratio* is the point.

## Beyond the core

- **Adaptive downshift router** (`router.py`, wired into `live_team` via `run --adaptive`) —
  tries the cheapest model on the writer subtask first and escalates up a ladder only when the
  validator rejects, recording every attempt. The active form of the SLM thesis (vs. the static
  cost *report*).
- **Correlated-failure mitigation** (`remediate.py`) — when agents fail together because they
  share a base model, `analyze --mitigate` proposes reassigning one to a different model.
- **Framework adapter** (`adapters/ingest.py`) — ingests a normalized event stream any
  LangGraph/CrewAI callback can emit, so the detectors/verifier/cost run over *external* agent
  systems too. A `FrameworkTracer` helper shows the wiring.
- **Static HTML dashboard** (`dashboard.py`) — a single self-contained file (no server, no JS
  deps) for trace replay, the root-cause banner, and the cost panel.

## Architecture

See **[SCOPE.md](SCOPE.md)** for the full design (AgentStep schema, error taxonomy, demo
script). All MVP and stretch items are implemented.

## Layout

```
src/
  schema.py              AgentStep / Plan / Subtask data model
  recorder.py            TraceRecorder (write) + Trace (load/replay)
  agents/team.py         scripted demo team (reproducible failure modes)
  agents/live_team.py    live team — real LLM calls, gated by the plan verifier
  agents/llm.py          Ollama + NIM/Nemotron backends (one interface)
  detectors/coordination.py   the error-taxonomy detectors
  verifier/fd_runner.py       thin Fast Downward runner
  verifier/plan_verifier.py   plan → PDDL → solve (the pre-execution gate)
  router.py              adaptive model-downshift router
  remediate.py           correlated-failure mitigation (model diversification)
  cost.py                SLM-routing cost model
  adapters/ingest.py     ingest external (LangGraph/CrewAI) event streams
  dashboard.py           standalone HTML trace viewer
  report.py / run.py / analyze.py / verify_plan.py / route_demo.py   CLIs
tests/                   pipeline + detector + routing tests (20)
pyproject.toml  Makefile  LICENSE  requirements.txt
```

## Status

Complete and runnable — recorder, **six** coordination-failure detectors, Fast Downward plan
gate, live Ollama/NIM backends, SLM-routing cost model, adaptive downshift router (wired into
the live loop), correlated-failure mitigation, framework adapter, and HTML dashboard. Verified
with reproducible scripted scenarios and a real local-model run (`make test` → **20 passing**).
Python ≥3.10; Fast Downward at `~/fast_downward/fast-downward.py`.
