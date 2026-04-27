"""Generic contract test (template). Copied verbatim into each case dir as
test_equivalence.py — adjust ATOL/RTOL inline per case if numerical precision
demands it.

Contract: jax_code.compute(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray].
The test loads frozen inputs.npz + expected.npz, calls compute(), and asserts
elementwise closeness on every output key.
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from _test_utils import assert_close

sys.path.insert(0, str(HERE))
from jax_code import compute


ATOL = 1e-5
RTOL = 1e-5
CASE = HERE.name


def main():
    inputs = dict(np.load(HERE / "inputs.npz"))
    expected = dict(np.load(HERE / "expected.npz"))
    actual = compute(inputs)

    missing = set(expected) - set(actual)
    extra = set(actual) - set(expected)
    if missing or extra:
        raise AssertionError(f"output key mismatch: missing={missing} extra={extra}")
    for k in expected:
        assert_close(np.asarray(actual[k]), expected[k], atol=ATOL, rtol=RTOL, name=k)
    print(f"[{CASE}] contract PASS")


if __name__ == "__main__":
    main()
