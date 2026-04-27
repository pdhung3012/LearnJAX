"""Freeze fixtures for m7: SimpleNN forward (28*28 -> 128 -> 10) with shared weights."""
import numpy as np
import torch
import torch.nn as nn


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "fc1_w": (rng.standard_normal((128, 28*28)) * 0.05).astype(np.float32),
        "fc1_b": (rng.standard_normal((128,)) * 0.05).astype(np.float32),
        "fc2_w": (rng.standard_normal((10, 128)) * 0.05).astype(np.float32),
        "fc2_b": (rng.standard_normal((10,)) * 0.05).astype(np.float32),
        "x":     rng.standard_normal((4, 1, 28, 28)).astype(np.float32),
    }


def pytorch_reference(inputs):
    fc1 = nn.Linear(28*28, 128); fc2 = nn.Linear(128, 10)
    with torch.no_grad():
        fc1.weight.copy_(torch.from_numpy(inputs["fc1_w"])); fc1.bias.copy_(torch.from_numpy(inputs["fc1_b"]))
        fc2.weight.copy_(torch.from_numpy(inputs["fc2_w"])); fc2.bias.copy_(torch.from_numpy(inputs["fc2_b"]))
    flat = torch.from_numpy(inputs["x"]).view(-1, 28*28)
    return {"logits": fc2(torch.relu(fc1(flat))).detach().numpy()}


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("m7: fixtures written")
