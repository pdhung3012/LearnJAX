"""Freeze fixtures for e5: 2->10->1 MLP forward (ReLU between)."""
import numpy as np
import torch
import torch.nn as nn


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "fc1_W": (rng.standard_normal((10, 2)) * 0.3).astype(np.float32),
        "fc1_b": (rng.standard_normal((10,)) * 0.1).astype(np.float32),
        "fc2_W": (rng.standard_normal((1, 10)) * 0.3).astype(np.float32),
        "fc2_b": (rng.standard_normal((1,)) * 0.1).astype(np.float32),
        "X":     rng.uniform(0, 10, (100, 2)).astype(np.float32),
    }


def pytorch_reference(inputs):
    fc1 = nn.Linear(2, 10); fc2 = nn.Linear(10, 1)
    with torch.no_grad():
        fc1.weight.copy_(torch.from_numpy(inputs["fc1_W"]))
        fc1.bias.copy_(torch.from_numpy(inputs["fc1_b"]))
        fc2.weight.copy_(torch.from_numpy(inputs["fc2_W"]))
        fc2.bias.copy_(torch.from_numpy(inputs["fc2_b"]))
    out = fc2(torch.relu(fc1(torch.from_numpy(inputs["X"]))))
    return {"predictions": out.detach().numpy()}


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("e5: fixtures written")
