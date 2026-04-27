"""Freeze fixtures for e3: Linear(1,1) -> tanh(z) + z forward."""
import numpy as np
import torch
import torch.nn as nn


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "W": rng.uniform(-1, 1, (1, 1)).astype(np.float32),
        "b": rng.uniform(-1, 1, (1,)).astype(np.float32),
        "X": rng.uniform(0, 10, (100, 1)).astype(np.float32),
    }


def pytorch_reference(inputs):
    layer = nn.Linear(1, 1)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(inputs["W"]))
        layer.bias.copy_(torch.from_numpy(inputs["b"]))
    z = layer(torch.from_numpy(inputs["X"]))
    return {"predictions": (torch.tanh(z) + z).detach().numpy()}


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("e3: fixtures written")
