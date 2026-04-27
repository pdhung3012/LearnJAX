"""Generate canonical inputs.npz and PyTorch-reference expected.npz for e1.

Run this once (and any time the canonical inputs change) to refresh the
golden fixtures. The fixtures are committed; tests load them directly.

The "case definition" lives here — pytorch_code.py is messy script-style code
intended as the *input* to the cheap LLM, not as the canonical reference.
"""
import numpy as np
import torch
import torch.nn as nn


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "W": rng.uniform(-1, 1, (1, 1)).astype(np.float32),  # PyTorch (out, in)
        "b": rng.uniform(-1, 1, (1,)).astype(np.float32),
        "X": rng.uniform(0, 10, (100, 1)).astype(np.float32),
    }


def pytorch_reference(inputs):
    """Reference PyTorch implementation: nn.Linear(1, 1) forward pass."""
    layer = nn.Linear(1, 1)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(inputs["W"]))
        layer.bias.copy_(torch.from_numpy(inputs["b"]))
    return {
        "predictions": layer(torch.from_numpy(inputs["X"])).detach().numpy()
    }


def main():
    inputs = make_inputs()
    expected = pytorch_reference(inputs)
    np.savez("inputs.npz",   **inputs)
    np.savez("expected.npz", **expected)
    print("e1: wrote inputs.npz and expected.npz")
    for k, v in inputs.items():
        print(f"  input[{k}]: shape={v.shape}, dtype={v.dtype}")
    for k, v in expected.items():
        print(f"  expected[{k}]: shape={v.shape}, dtype={v.dtype}")


if __name__ == "__main__":
    main()
