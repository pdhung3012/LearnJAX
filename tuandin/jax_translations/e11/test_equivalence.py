"""e11 contract test: BPE merges/vocab match exactly (JSON string compare).

The contract is the same as the other cases (compute(inputs) -> dict), but
the outputs are JSON-encoded structures, not numpy tensors, so we compare
strings rather than numerical closeness.
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from jax_code import compute


def main():
    inputs = dict(np.load(HERE / "inputs.npz", allow_pickle=True))
    expected = dict(np.load(HERE / "expected.npz", allow_pickle=True))
    actual = compute(inputs)

    missing = set(expected) - set(actual)
    extra = set(actual) - set(expected)
    if missing or extra:
        raise AssertionError(f"output key mismatch: missing={missing} extra={extra}")
    for k in expected:
        a = str(actual[k]); e = str(expected[k])
        if a != e:
            raise AssertionError(f"{k}: differs\n  actual:   {a}\n  expected: {e}")
        print(f"  {k}: ✓")
    print("[e11] contract PASS")


if __name__ == "__main__":
    main()
