"""Generic freeze_fixtures (template). Copied verbatim into each case dir.

Each case's pytorch_code.py exposes:
  - make_inputs() -> dict[str, np.ndarray]   # canonical inputs
  - compute(inputs) -> dict[str, np.ndarray] # PyTorch reference implementation
This script runs them, saves inputs.npz + expected.npz.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from pytorch_code import compute, make_inputs


def main():
    inputs = make_inputs()
    expected = compute(inputs)
    np.savez(HERE / "inputs.npz", **inputs)
    np.savez(HERE / "expected.npz", **expected)
    print(f"{HERE.name}: fixtures written")


if __name__ == "__main__":
    main()
