# nemoclaw-agent

> A reliability layer for multi-agent systems — a "flight recorder" + formal plan verifier
> that sits *over* any agent framework, catches coordination failures, and can block a bad
> plan **before it runs** using a classical planner.

## The pitch (one paragraph)

Everyone is building teams of specialist agents (LangGraph/CrewAI swarms — a ~1,445% surge in
multi-agent inquiries through 2025). But the research consensus is that multi-agent
**reliability and correctness is underexplored**: errors propagate between agents, agents on a
shared base model fail in correlated ways, and there is no standard way to "formalize
coordination logic, arbitration, and failure handling." nemoclaw-agent is an open take on
NVIDIA's **NemoClaw** (the orchestration + policy-enforcement layer shipped at GTC 2026): it
instruments an existing agent team, records a structured execution trace, detects coordination
failures from that trace, and — the showpiece — formally verifies the planner's proposed plan
with **Fast Downward** before execution, blocking goals that are provably unreachable.

## Why this project (differentiation)

- **Neuro-symbolic moat.** Reuses the Nemotron → PDDL → Fast Downward verification pipeline
  already built in `../av-scenario-forge`. The multi-agent crowd cannot do formal plan
  verification because they have no classical planner wired up. We do.
- **Targets the named gap.** Coordination failures / error propagation / correlated failures /
  execution-trace analysis are exactly what the field admits it has no good answer for.
- **A layer, not a framework.** It wraps existing agents (mirrors how NVIDIA's NeMo Agent
  Toolkit onboards existing LangGraph agents) instead of competing with them — less work,
  better story.
- **On-brand by name.** Real NemoClaw = "orchestration + policy-enforcement layer." This is an
  open implementation of that category, a current and credible interview talking point.

## Architecture

```
        existing agent team (planner → workers → critic)
                          │  (instrumented via callbacks/shim)
                          ▼
            ┌──────────────────────────────┐
   (a)      │   Pre-execution Plan Verifier │  ← Fast Downward (reuse av-scenario-forge)
            │   plan → PDDL → solve         │     blocks provably-unreachable goals
            └───────────────┬──────────────┘
                            │ plan OK → run
                            ▼
            ┌──────────────────────────────┐
   (b)      │      Trace Recorder           │  → JSONL of every AgentStep
            │  (the "flight recorder")      │
            └───────────────┬──────────────┘
                            │ trace
                            ▼
            ┌──────────────────────────────┐
   (c)      │  Coordination-Failure Detector│  → error taxonomy + post-mortem report
            │  + correlated-failure check   │     "agent B broke the chain because…"
            └──────────────────────────────┘
```

### (a) Pre-execution plan verifier — the showpiece
The planner agent emits an ordered plan: subtasks with preconditions/effects and a goal.
Compile that to a PDDL problem against a small coordination domain and run Fast Downward.
- Plan found → goal is reachable, ordering is consistent → allow execution.
- No plan → goal unreachable / precondition unsatisfiable / ordering conflict → **block** and
  report which subtask/precondition is the culprit.
Reuse the `CoverageChecker` pattern from `../av-scenario-forge/src/verifier/coverage_checker.py`
(subprocess call to `~/fast_downward/fast-downward.py`). Factor the FD-runner into a thin
shared core.

### (b) Trace recorder
Append-only JSONL. One record per step:
```jsonc
{
  "step_id": 7,
  "agent_id": "writer",
  "role": "worker",
  "model": "qwen2.5:3b",            // enables correlated-failure analysis
  "ts_start": 0.0, "ts_end": 4.2,
  "input_refs": ["step_5.output"],  // provenance → error-propagation tracking
  "tool_calls": [{"name": "search", "args": {...}, "ok": true}],
  "output": "...",
  "status": "ok | error | timeout",
  "claims_satisfied": ["draft_written"]  // postconditions the step asserts
}
```

### (c) Coordination-failure detector (the error taxonomy)
Post-hoc analysis over the trace. **All six categories below are implemented**
(`src/detectors/coordination.py`), plus a `goal_unreached` symptom that never
headlines over a real root cause:
| Category | Detection signal |
|----------|------------------|
| Deadlock / livelock | repeated state with no new `claims_satisfied`; step budget exceeded |
| Conflicting actions | two agents assert contradictory claims about shared state |
| Error propagation | a step with `status:error`/bad output is in another step's `input_refs` chain |
| Correlated failure | steps sharing `model` fail on the same subtask → flag base-model risk |
| Dropped handoff | a required precondition never produced by any upstream step |
| Schema/contract violation | tool args or output fail the declared schema |

## MVP vs. stretch

**MVP (build first):**
1. A minimal 3-agent demo task (planner → 2 workers → critic) — no heavy framework; a small
   hand-rolled loop so the trace format is fully under our control.
2. Trace recorder (JSONL) + the AgentStep schema above.
3. Plan verifier on a small coordination PDDL domain (reuse FD runner).
4. 4 detectors: deadlock, error propagation, dropped handoff, correlated failure.
5. CLI post-mortem report (use `rich`): timeline + the single root-cause line.

**Stretch (all implemented):**
- [x] Live LLM backends — Ollama (local SLMs) + NIM/Nemotron, one interface (`agents/llm.py`,
  `agents/live_team.py`); live runs are gated by the plan verifier.
- [x] Cost-aware SLM routing report (`cost.py`): routed cost vs all-frontier counterfactual.
- [x] Per-worker *adaptive* downshift router (`router.py`), **wired into the live loop**
  (`run --adaptive`) and demoable offline (`route_demo.py`): try the cheapest model first,
  escalate up a ladder only when the validator rejects; every attempt is recorded.
- [x] Framework adapter (`adapters/ingest.py`): ingest a normalized event stream that any
  LangGraph/CrewAI callback can emit → same detectors/verifier/cost apply. "Layer over any framework."
- [x] Static HTML dashboard (`dashboard.py`): single self-contained file, trace replay + findings + cost.
- [x] Correlated-failure *mitigation* (`remediate.py`): auto-diversify the shared model for
  agents that failed together (`analyze --mitigate`).

## Demo script (make it memorable)
1. Run a 3-agent task that **silently goes wrong** (worker B consumes worker A's bad output).
2. Replay the JSONL through the recorder → report pinpoints **which** agent broke the chain
   and **why** (error-propagation path).
3. Showpiece: feed the planner a goal whose preconditions can't be met → Fast Downward
   **blocks it before any agent runs**, naming the unsatisfiable precondition.

## Tech stack
- Python 3.12 (`.venv` like the sibling projects).
- Reuse from `../av-scenario-forge`: `NIMClient` (Nemotron 3 Super via NIM, `NIM_API_KEY`) for
  the planner/critic; `CoverageChecker`/FD-runner pattern for verification.
- Ollama small models (e.g. `qwen2.5:3b`, already pinned locally) for worker agents → local,
  CPU-only, cheap, and sets up the correlated-failure + SLM-routing angles.
- Fast Downward at `~/fast_downward/fast-downward.py`.
- JSONL traces; `rich` for the CLI report.

## Open design questions (decide before coding the verifier)
- Coordination PDDL domain: how rich? Start minimal (subtasks as actions, claims as predicates,
  goal = critic-acceptance) and grow only if a detector needs it.
- Demo task domain: research→write→review is generic and easy to "break" on purpose — good
  default unless we want something more NVIDIA-flavored (e.g. an AV-incident triage pipeline,
  which would rhyme with av-scenario-forge).
