"""
LLM backends for live agent runs.

Two backends, one interface (`LLM.chat -> Reply`):
  - OllamaLLM  : local models via http://localhost:11434 (no extra pip dep)
  - NIMLLM     : NVIDIA NIM / Nemotron via the openai client (needs NIM_API_KEY)

Model specs are strings: "ollama:qwen2.5:3b" or "nim:nvidia/nemotron-3-super-120b-a12b".
The scripted backend (src/agents/team.py) needs none of this.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


@dataclass
class Reply:
    text: str
    latency_s: float
    tokens: int          # output tokens (eval_count / completion_tokens)
    model: str


class LLM:
    name: str            # short label recorded in the trace
    def chat(self, messages: list[dict], num_predict: int = 256,
             temperature: float = 0.2) -> Reply: ...


class OllamaLLM(LLM):
    def __init__(self, model: str):
        self.model = model
        self.name = model

    def chat(self, messages, num_predict: int = 256, temperature: float = 0.2) -> Reply:
        payload = {
            "model": self.model, "messages": messages, "stream": False,
            "options": {"num_predict": num_predict, "temperature": temperature},
        }
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        t = time.monotonic()
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        return Reply(r["message"]["content"], time.monotonic() - t,
                     int(r.get("eval_count", 0)), self.model)


class NIMLLM(LLM):
    def __init__(self, model: str):
        from openai import OpenAI
        key = os.environ.get("NIM_API_KEY")
        if not key:
            raise ValueError("NIM_API_KEY not set (get a free key at https://build.nvidia.com)")
        self.model = model
        self.name = model.split("/")[-1]
        self._client = OpenAI(base_url=NIM_BASE_URL, api_key=key)

    def chat(self, messages, num_predict: int = 256, temperature: float = 0.2) -> Reply:
        t = time.monotonic()
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=num_predict)
        usage = getattr(resp, "usage", None)
        toks = getattr(usage, "completion_tokens", 0) if usage else 0
        return Reply(resp.choices[0].message.content, time.monotonic() - t,
                     toks, self.name)


def make_llm(spec: str) -> LLM:
    """spec = 'ollama:<model>' or 'nim:<model>'. Bare names default to ollama."""
    if spec.startswith("nim:"):
        return NIMLLM(spec[4:])
    if spec.startswith("ollama:"):
        return OllamaLLM(spec[7:])
    return OllamaLLM(spec)
