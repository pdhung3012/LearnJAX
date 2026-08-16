"""Prompt templates for translator and fixer roles."""

import numpy as np
from pathlib import Path


def describe_schema(case_dir: Path) -> str:
    """Human-readable schema derived from inputs.npz + expected.npz.

    The eval harness tests a pure-function contract `compute(inputs) -> dict`,
    not the full training loop. The LLM cannot infer this contract from
    pytorch_code.py alone — we must state it explicitly.
    """
    inputs = dict(np.load(case_dir / "inputs.npz"))
    expected = dict(np.load(case_dir / "expected.npz"))

    def _fmt(d):
        return "\n".join(
            f"    {k!r}: shape={v.shape} dtype={v.dtype}" for k, v in d.items()
        )

    return (
        f"Contract for `compute(inputs) -> dict`:\n"
        f"  Input keys (numpy arrays):\n{_fmt(inputs)}\n"
        f"  Required output keys (numpy-convertible):\n{_fmt(expected)}"
    )


SYSTEM_TRANSLATE = """You are an expert in PyTorch and JAX/Flax.

The evaluation is a PURE-FUNCTION equivalence test, NOT a training loop replay.
You must produce a JAX file that exposes exactly one function:

    def compute(inputs: dict) -> dict:
        ...

Rules:
- Read inputs from the `inputs` dict — do NOT generate synthetic data, do NOT
  re-seed RNGs, do NOT re-run training. The inputs already contain any
  trained weights / preprocessed data you need.
- Return a dict with EXACTLY the output keys listed in the schema below.
- Use JAX/Flax idioms (jax.numpy, flax.linen, jax.random keys) where relevant.
- No test code, no `if __name__ == "__main__"`, no prints.

Return ONLY the code inside a single ```python fenced block. No prose."""

USER_TRANSLATE = """PyTorch reference (may include training code you should NOT replicate — only translate the forward-compute logic that produces the required outputs):

```python
{pytorch_code}
```

{schema}

Implement `compute(inputs) -> dict` in JAX/Flax matching the contract above."""


SYSTEM_FIX = """You are an expert JAX/Flax debugger. Given a PyTorch reference, a
buggy JAX translation, the contract schema, and the executable oracle's error
message, produce a corrected JAX file.

Rules:
- Output MUST expose `compute(inputs: dict) -> dict` with EXACTLY the schema's
  input reads and output keys.
- Do NOT generate synthetic data or re-train; read weights/inputs from
  `inputs`.
- Fix the specific error reported by the oracle.
- Return ONLY the corrected code inside a single ```python fenced block. No
  prose."""

USER_FIX = """PyTorch reference:

```python
{pytorch_code}
```

{schema}

Current (buggy) JAX translation:

```python
{jax_code}
```

Oracle status: {status}
Oracle error:
{error}

{diff_hint}

Produce a corrected JAX file."""


def diff_hint(max_diff: float | None, per_output: dict[str, float]) -> str:
    if max_diff is None:
        return ""
    parts = [f"Numerical diffs (max={max_diff:.2e}):"]
    for k, v in sorted(per_output.items(), key=lambda x: -x[1])[:5]:
        parts.append(f"  {k}: {v:.2e}")
    return "\n".join(parts)


def extract_code(text: str | None) -> str:
    """Pull the first ```python ... ``` block; fall back to whole text.

    Robust to None / empty replies (some providers emit reasoning-only turns
    with no content).
    """
    if not text:
        return ""
    import re
    m = re.search(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()
