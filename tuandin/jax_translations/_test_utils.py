"""Shared helpers for test_equivalence.py scripts.

Two patterns are supported:

1. **Direct equivalence test** — re-build the algorithmic core in both frameworks
   from shared inputs/weights and compare numerically. This is the right thing
   to do for deterministic forward passes; PyTorch and JAX have different RNGs,
   so comparing whole training runs is not meaningful.

2. **Smoke test** — when neither weight transfer nor a deterministic comparison
   is practical (dataset downloads, GUI plots, training loops with different
   RNG), simply run both scripts as subprocesses and assert both return 0.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PYTHON = "/opt/miniconda3/envs/t2j/bin/python"


def run_script(script_path: Path, timeout: int = 300) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"  # don't pop GUI windows.
    return subprocess.run(
        [PYTHON, str(script_path)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(script_path.parent),
        timeout=timeout,
    )


def smoke_test(here: Path, name: str, timeout: int = 300) -> None:
    """Run pytorch_code.py and jax_code.py; assert both succeed."""
    for fname in ("pytorch_code.py", "jax_code.py"):
        path = here / fname
        if not path.exists():
            print(f"[{name}] {fname} missing — skip")
            continue
        proc = run_script(path, timeout=timeout)
        if proc.returncode != 0:
            print(f"[{name}] {fname} STDERR (last 1500 chars):")
            print(proc.stderr[-1500:])
            raise AssertionError(f"{fname} returned {proc.returncode}")
        print(f"[{name}] {fname} ✓ (last stdout line: "
              f"{proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else '<empty>'})")
    print(f"[{name}] smoke test PASS")


def assert_close(a, b, *, atol: float = 1e-5, rtol: float = 1e-5, name: str = "tensor") -> None:
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    diff = np.abs(a - b).max()
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{name}: max abs diff = {diff:.3e} (atol={atol}, rtol={rtol})"
        )
    print(f"  {name}: ✓ max abs diff = {diff:.3e} (shape {a.shape})")


def torch_linear_to_jax(weight, bias):
    """Convert a torch nn.Linear's weight (out, in) and bias (out,)
    to JAX kernel (in, out) and bias (out,)."""
    import jax.numpy as jnp
    return jnp.asarray(weight.detach().numpy().T), jnp.asarray(bias.detach().numpy())
