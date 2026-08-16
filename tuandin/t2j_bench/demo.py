"""Standalone OpenRouter demo — safe to hand to a colleague.

Usage:
    OPEN_ROUTER_KEY=sk-or-... python -m tuandin.t2j_bench.demo

    # or, if running the file directly:
    OPEN_ROUTER_KEY=sk-or-... python tuandin/t2j_bench/demo.py

This script has ZERO dependencies beyond the Python standard library.
It makes one call for each of the three pipeline roles:

    1. WEAK translator  — cheap open-weight model, translates a tiny
       PyTorch snippet to JAX.
    2. STRONG fixer     — mid-tier frontier model, given the same snippet
       plus an intentionally-broken JAX attempt + error, produces a fix.
    3. RC judge         — cheap frontier-family model, scores whether the
       fixed JAX is a faithful translation.

Each call prints: latency, prompt/completion tokens, and USD cost using
the pricing table baked into this file. Total spend for one full demo
run is well under 1 cent.

Adapt this file as a starting point for your own experiments. Nothing
here mandates the full T2J-Bench pipeline — it's just an auth + call
example.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

API_URL = "https://openrouter.ai/api/v1/chat/completions"
ENV_VAR = "OPEN_ROUTER_KEY"


# --------------------------------------------------------------------------
# Pricing table (USD per 1M tokens). Snapshot: 2026-08-15.
# Refresh from openrouter.ai/models before large runs.
# --------------------------------------------------------------------------
PRICING = {
    # Weak / open-weight translators
    "ibm-granite/granite-4.1-8b":          (0.05, 0.10),
    "meta-llama/llama-3.1-8b-instruct":    (0.05, 0.08),
    "google/gemma-3-12b-it":               (0.05, 0.15),
    "microsoft/phi-4":                     (0.07, 0.14),
    "openai/gpt-oss-20b":                  (0.03, 0.13),
    "qwen/qwen-2.5-7b-instruct":           (0.10, 0.20),
    "mistralai/codestral-2508":            (0.30, 0.90),

    # Strong fixers
    "z-ai/glm-5.2":                        (0.46, 1.45),
    "moonshotai/kimi-k2.7-code":           (0.71, 3.50),
    "deepseek/deepseek-v4-pro":            (1.17, 2.34),
    "google/gemini-2.5-pro":               (1.25, 10.00),
    "x-ai/grok-4.6":                       (2.00, 6.00),
    "openai/gpt-5.4":                      (2.50, 15.00),
    "anthropic/claude-sonnet-4.6":         (3.00, 15.00),
    "anthropic/claude-opus-4.8":           (5.00, 25.00),

    # Judges
    "google/gemini-2.5-flash":             (0.30, 2.50),
    "openai/gpt-5-mini":                   (0.25, 2.00),
    "google/gemini-3.5-flash":             (1.50, 9.00),
}


def load_key() -> str:
    key = os.environ.get(ENV_VAR, "").strip()
    if not key:
        sys.exit(
            f"error: {ENV_VAR} not set.\n"
            f"usage: {ENV_VAR}=sk-or-... python -m tuandin.t2j_bench.demo"
        )
    return key


def chat(
    key: str,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: float = 120.0,
) -> dict:
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # Optional but polite: identify your app to OpenRouter.
            "HTTP-Referer": "https://github.com/tuanad121/LearnJAX",
            "X-Title": "T2J-Bench-demo",
        },
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} from OpenRouter: {e.read().decode()[:400]}")
    latency = time.time() - t0

    msg = data["choices"][0].get("message", {}) or {}
    text = msg.get("content") or msg.get("reasoning") or ""
    usage = data.get("usage", {})
    tin = int(usage.get("prompt_tokens", 0))
    tout = int(usage.get("completion_tokens", 0))
    pin, pout = PRICING.get(model, (0.0, 0.0))
    cost = (tin * pin + tout * pout) / 1e6
    return {
        "text": text,
        "latency_s": latency,
        "tokens_in": tin,
        "tokens_out": tout,
        "cost": cost,
    }


def announce(role: str, model: str) -> None:
    pin, pout = PRICING.get(model, (0.0, 0.0))
    print(f"\n--- {role} · {model} (${pin:.2f}/${pout:.2f} per 1M) ---")


def show(result: dict, *, tail_chars: int = 400) -> None:
    text = result["text"] or "<empty response>"
    print(f"latency: {result['latency_s']:.2f}s | "
          f"in={result['tokens_in']} out={result['tokens_out']} tokens | "
          f"cost=${result['cost']:.6f}")
    print("--- response tail ---")
    print(text[-tail_chars:])


PYTORCH_SNIPPET = """
import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
    def forward(self, x):
        return torch.relu(self.fc(x))
""".strip()

BROKEN_JAX = """
import jax.numpy as jnp
from flax import linen as nn

class TinyNet(nn.Module):
    dim: int
    @nn.compact
    def __call__(self, x):
        # BUG: wrong activation, and missing Dense feature count
        return jnp.tanh(nn.Dense()(x))
""".strip()

ORACLE_ERROR = (
    "TypeError: nn.Dense() missing required argument 'features'; "
    "and behaviourally: PyTorch uses ReLU but this JAX uses tanh."
)


def role_weak_translator(key: str) -> dict:
    model = "ibm-granite/granite-4.1-8b"   # cheapest weak translator
    announce("WEAK translator", model)
    return chat(key, model, [
        {"role": "system", "content":
         "You are an expert PyTorch→JAX/Flax translator. Return only Python code."},
        {"role": "user", "content":
         f"Translate this PyTorch module to JAX/Flax:\n\n```python\n{PYTORCH_SNIPPET}\n```"},
    ], temperature=0.7)


def role_strong_fixer(key: str) -> dict:
    model = "z-ai/glm-5.2"   # cheap strong fixer for demo
    announce("STRONG fixer", model)
    return chat(key, model, [
        {"role": "system", "content":
         "You are an expert JAX/Flax debugger. Return only Python code."},
        {"role": "user", "content":
         f"PyTorch reference:\n```python\n{PYTORCH_SNIPPET}\n```\n\n"
         f"Broken JAX attempt:\n```python\n{BROKEN_JAX}\n```\n\n"
         f"Oracle error: {ORACLE_ERROR}\n\nProduce a corrected JAX file."},
    ], temperature=0.3)


def role_rc_judge(key: str, jax_code: str) -> dict:
    model = "google/gemini-2.5-flash"   # cheap third-family judge
    announce("RC judge", model)
    return chat(key, model, [
        {"role": "system", "content":
         "You evaluate readability and faithfulness of JAX translations of "
         "PyTorch code. Reply with JSON: {\"faithful\": true|false, "
         "\"idiomatic\": true|false, \"notes\": \"...\"}. No prose."},
        {"role": "user", "content":
         f"PyTorch:\n```python\n{PYTORCH_SNIPPET}\n```\n\n"
         f"JAX:\n```python\n{jax_code[:1500]}\n```"},
    ], temperature=0.0)


def main() -> None:
    key = load_key()
    total = 0.0

    r1 = role_weak_translator(key)
    show(r1)
    total += r1["cost"]

    r2 = role_strong_fixer(key)
    show(r2)
    total += r2["cost"]

    r3 = role_rc_judge(key, r2["text"] or r1["text"])
    show(r3)
    total += r3["cost"]

    print(f"\n=== total demo cost: ${total:.6f} ===")


if __name__ == "__main__":
    main()
