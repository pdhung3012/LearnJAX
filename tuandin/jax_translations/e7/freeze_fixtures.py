"""Freeze fixtures for e7: Linear(1,1) forward (save/load tested by integration script)."""
import numpy as np
import torch
import torch.nn as nn


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "W": rng.uniform(-1, 1, (1, 1)).astype(np.float32),
        "b": rng.uniform(-1, 1, (1,)).astype(np.float32),
        "X": rng.uniform(0, 1, (3, 1)).astype(np.float32),
    }


def pytorch_reference(inputs):
    layer = nn.Linear(1, 1)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(inputs["W"]))
        layer.bias.copy_(torch.from_numpy(inputs["b"]))
    return {"predictions": layer(torch.from_numpy(inputs["X"])).detach().numpy()}


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("e7: fixtures written")
