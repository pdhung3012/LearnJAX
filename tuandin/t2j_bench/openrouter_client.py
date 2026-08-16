"""Thin OpenRouter chat-completion wrapper with cost tracking.

Key is read from tuandin/open_router.txt (single line, sk-or-... form).
Never printed or logged.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path

from .models import Model


API_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_FILE = Path(__file__).resolve().parent.parent / "open_router.txt"
ENV_VAR = "OPEN_ROUTER_KEY"


def load_key() -> str:
    """Return the OpenRouter key.

    Resolution order:
      1. OPEN_ROUTER_KEY env var (colleague workflow)
      2. tuandin/open_router.txt file (principal workflow)
    """
    env = os.environ.get(ENV_VAR)
    if env and env.strip():
        return env.strip()
    if KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
        if key:
            return key
    raise RuntimeError(
        f"No OpenRouter key found. Set the {ENV_VAR} env var, "
        f"or write it to {KEY_FILE}."
    )


@dataclass
class CallResult:
    text: str
    tokens_in: int
    tokens_out: int
    cost: float
    latency_s: float


@dataclass
class CostLedger:
    total_cost: float = 0.0
    calls: int = 0
    per_model: dict[str, float] = field(default_factory=dict)

    def add(self, model: Model, cost: float) -> None:
        self.total_cost += cost
        self.calls += 1
        self.per_model[model.slug] = self.per_model.get(model.slug, 0.0) + cost

    def summary(self) -> str:
        lines = [f"total: ${self.total_cost:.4f} across {self.calls} calls"]
        for slug, c in sorted(self.per_model.items(), key=lambda x: -x[1]):
            lines.append(f"  {slug:<45} ${c:.4f}")
        return "\n".join(lines)


class OpenRouterClient:
    def __init__(self, ledger: CostLedger | None = None, timeout: float = 120.0):
        self._key = load_key()
        self.ledger = ledger or CostLedger()
        self.timeout = timeout

    def chat(
        self,
        model: Model,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        retries: int = 3,
    ) -> CallResult:
        payload = {
            "model": model.slug,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/tuanad121/LearnJAX",
                "X-Title": "T2J-Bench",
            },
        )

        backoff = 2.0
        for attempt in range(retries):
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                latency = time.time() - t0
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                if attempt == retries - 1:
                    raise RuntimeError(f"openrouter call failed after {retries} tries: {e}")
                time.sleep(backoff)
                backoff *= 2
        else:
            raise RuntimeError("unreachable")

        msg = data["choices"][0].get("message", {}) or {}
        choice = msg.get("content") or msg.get("reasoning") or ""
        if not isinstance(choice, str):
            # Some providers return content as list of parts
            try:
                choice = "".join(
                    p.get("text", "") for p in choice if isinstance(p, dict)
                )
            except Exception:
                choice = str(choice)
        usage = data.get("usage", {})
        tin = int(usage.get("prompt_tokens", 0))
        tout = int(usage.get("completion_tokens", 0))
        cost = model.estimate_cost(tin, tout)
        self.ledger.add(model, cost)
        return CallResult(text=choice, tokens_in=tin, tokens_out=tout, cost=cost, latency_s=latency)
