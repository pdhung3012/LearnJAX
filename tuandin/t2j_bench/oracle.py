"""Executable oracle for PyTorch→JAX equivalence.

Consumes a case directory laid out per tuandin/jax_translations/*/ convention:

    case_dir/
        pytorch_code.py   # ground truth
        jax_code.py       # candidate translation (we write this)
        inputs.npz        # inputs to compute()
        expected.npz      # reference outputs (from pytorch_code)
        test_equivalence.py  # existing contract test (unused; we run our own)

The candidate jax_code.py must expose `compute(inputs: dict) -> dict`.
We run the check in a subprocess so translator/fixer output that segfaults
or hangs cannot take down the pipeline.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ATOL = 1e-5
RTOL = 1e-5
TIMEOUT_S = 120


# We invoke this Python one-liner in a subprocess to isolate candidate code.
# It prints a JSON result blob on stdout that the parent parses.
_RUNNER = r"""
import json, sys, traceback, numpy as np
from pathlib import Path
HERE = Path(sys.argv[1])
sys.path.insert(0, str(HERE))
result = {"status": "UNKNOWN", "error": None, "max_diff": None, "per_output": {}}
try:
    import jax_code
except Exception as e:
    result["status"] = "IMPORT_ERROR"
    result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
    print(json.dumps(result)); sys.exit(0)
if not hasattr(jax_code, "compute"):
    result["status"] = "CONTRACT_ERROR"
    result["error"] = "jax_code has no `compute` symbol"
    print(json.dumps(result)); sys.exit(0)
try:
    inputs = dict(np.load(HERE / "inputs.npz"))
    expected = dict(np.load(HERE / "expected.npz"))
    actual = jax_code.compute(inputs)
except Exception as e:
    result["status"] = "RUNTIME_ERROR"
    result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
    print(json.dumps(result)); sys.exit(0)
missing = sorted(set(expected) - set(actual))
extra = sorted(set(actual) - set(expected))
if missing or extra:
    result["status"] = "KEY_MISMATCH"
    result["error"] = f"missing={missing} extra={extra}"
    print(json.dumps(result)); sys.exit(0)
mx = 0.0
per = {}
try:
    for k in expected:
        a = np.asarray(actual[k]); b = np.asarray(expected[k])
        if a.shape != b.shape:
            result["status"] = "SHAPE_MISMATCH"
            result["error"] = f"{k}: {a.shape} vs {b.shape}"
            print(json.dumps(result)); sys.exit(0)
        d = float(np.abs(a - b).max())
        per[k] = d
        mx = max(mx, d)
    result["max_diff"] = mx
    result["per_output"] = per
    ATOL, RTOL = %(atol)r, %(rtol)r
    ok = all(np.allclose(np.asarray(actual[k]), expected[k], atol=ATOL, rtol=RTOL) for k in expected)
    result["status"] = "PASS" if ok else "NUMERIC_MISMATCH"
except Exception as e:
    result["status"] = "COMPARE_ERROR"
    result["error"] = f"{type(e).__name__}: {e}"
print(json.dumps(result))
""" % {"atol": ATOL, "rtol": RTOL}


@dataclass
class OracleResult:
    passed: bool
    status: str          # PASS / IMPORT_ERROR / RUNTIME_ERROR / KEY_MISMATCH / ...
    error: str | None    # short traceback tail (for fixer prompt)
    max_diff: float | None
    per_output: dict[str, float]

    def summary(self) -> str:
        if self.passed:
            return f"PASS (max_diff={self.max_diff:.2e})"
        return f"{self.status}: {self.error or '(no message)'}"


def run(case_dir: Path, python: str | None = None) -> OracleResult:
    """Execute case_dir/jax_code.py against case_dir/{inputs,expected}.npz."""
    python = python or _default_python()
    proc = subprocess.run(
        [python, "-c", _RUNNER, str(case_dir)],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    if proc.returncode != 0:
        return OracleResult(
            passed=False,
            status="SUBPROCESS_CRASH",
            error=(proc.stderr or proc.stdout)[-1500:],
            max_diff=None,
            per_output={},
        )
    try:
        blob = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as e:
        return OracleResult(
            passed=False,
            status="RUNNER_ERROR",
            error=f"could not parse runner output: {e}\nstdout tail: {proc.stdout[-500:]}",
            max_diff=None,
            per_output={},
        )
    return OracleResult(
        passed=blob["status"] == "PASS",
        status=blob["status"],
        error=blob.get("error"),
        max_diff=blob.get("max_diff"),
        per_output=blob.get("per_output") or {},
    )


def _default_python() -> str:
    # Match tuandin/jax_translations/_test_utils.py convention.
    candidate = "/opt/miniconda3/envs/t2j/bin/python"
    if Path(candidate).exists():
        return candidate
    return sys.executable
