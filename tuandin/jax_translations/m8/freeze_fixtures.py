"""Freeze fixtures for m8: encoder forward (Conv -> ReLU -> Pool x2)."""
import numpy as np
import torch
import torch.nn as nn


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "c1_w": (rng.standard_normal((32, 1, 3, 3)) * 0.1).astype(np.float32),
        "c1_b": (rng.standard_normal((32,)) * 0.05).astype(np.float32),
        "c2_w": (rng.standard_normal((64, 32, 3, 3)) * 0.1).astype(np.float32),
        "c2_b": (rng.standard_normal((64,)) * 0.05).astype(np.float32),
        "x":    rng.standard_normal((2, 1, 28, 28)).astype(np.float32),  # NCHW
    }


def pytorch_reference(inputs):
    c1 = nn.Conv2d(1, 32, 3, padding=1); c2 = nn.Conv2d(32, 64, 3, padding=1)
    pool = nn.MaxPool2d(2, 2)
    with torch.no_grad():
        c1.weight.copy_(torch.from_numpy(inputs["c1_w"])); c1.bias.copy_(torch.from_numpy(inputs["c1_b"]))
        c2.weight.copy_(torch.from_numpy(inputs["c2_w"])); c2.bias.copy_(torch.from_numpy(inputs["c2_b"]))
    h = pool(torch.relu(c1(torch.from_numpy(inputs["x"]))))
    h = pool(torch.relu(c2(h)))
    return {"encoded": h.detach().numpy()}


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("m8: fixtures written")
