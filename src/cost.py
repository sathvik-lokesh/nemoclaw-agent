"""
Cost model for the SLM-routing angle.

NVIDIA's thesis ("Small Language Models are the Future of Agentic AI") is that
most agent subtasks don't need a frontier model — routing workers to SLMs cuts
cost sharply at comparable task success. This module estimates the cost of a
recorded run and the counterfactual cost had every step used a frontier model.

Rates are ILLUSTRATIVE (USD per 1M output tokens) — adjust `RATES` to your
provider. The headline is the *ratio*, which is robust to the absolute numbers.
"""

from __future__ import annotations

from src.recorder import Trace

# Illustrative $/1M output tokens. Self-hosted SLMs priced at rough compute/electricity.
RATES = {
    "frontier": 3.00,     # e.g. a hosted Nemotron-Super-class model
    "slm": 0.06,          # locally-hosted 3B-class model (amortised compute)
}

# Map a model name to a tier.
def tier_of(model: str) -> str:
    m = model.lower()
    if "nemotron" in m or "frontier" in m or "120b" in m or "super" in m:
        return "frontier"
    return "slm"            # qwen2.5:3b, llama3.2:3b, phi3, scripted, etc.


def step_cost(model: str, tokens: int) -> float:
    return tokens / 1_000_000 * RATES[tier_of(model)]


def summarize(trace: Trace) -> dict:
    """Actual routed cost vs an all-frontier counterfactual."""
    actual = 0.0
    counterfactual = 0.0
    total_tokens = 0
    total_latency = 0.0
    per_model: dict[str, dict] = {}

    for s in trace.steps:
        # tokens are stored in the step note as "tokens=N" by the live team;
        # fall back to a length estimate when absent (scripted traces).
        toks = _tokens_of(s)
        total_tokens += toks
        total_latency += max(0.0, s.ts_end - s.ts_start)
        actual += step_cost(s.model, toks)
        counterfactual += step_cost("frontier", toks)
        pm = per_model.setdefault(s.model, {"steps": 0, "tokens": 0, "cost": 0.0})
        pm["steps"] += 1
        pm["tokens"] += toks
        pm["cost"] += step_cost(s.model, toks)

    savings = (1 - actual / counterfactual) * 100 if counterfactual else 0.0
    return {
        "actual_usd": actual,
        "all_frontier_usd": counterfactual,
        "savings_pct": savings,
        "total_tokens": total_tokens,
        "total_latency_s": total_latency,
        "per_model": per_model,
    }


def _tokens_of(step) -> int:
    for tok in step.note.split():
        if tok.startswith("tokens="):
            try:
                return int(tok.split("=", 1)[1])
            except ValueError:
                return 0
    # scripted fallback: ~4 chars/token of output
    return max(1, len(step.output) // 4)
