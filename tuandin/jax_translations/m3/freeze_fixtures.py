"""Freeze fixtures for m3: VanillaCNN forward (Conv->ReLU->Conv->ReLU->Pool->FC->ReLU->FC).

We test the conv-net forward pass, not training. Caller-supplied weights so RNG
init mismatch between PyTorch and JAX is irrelevant.
"""
import numpy as np
import torch
import torch.nn as nn


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "c1_w": (rng.standard_normal((32, 3, 3, 3)) * 0.1).astype(np.float32),
        "c1_b": (rng.standard_normal((32,)) * 0.05).astype(np.float32),
        "c2_w": (rng.standard_normal((64, 32, 3, 3)) * 0.1).astype(np.float32),
        "c2_b": (rng.standard_normal((64,)) * 0.05).astype(np.float32),
        "f1_w": (rng.standard_normal((128, 64*16*16)) * 0.01).astype(np.float32),
        "f1_b": (rng.standard_normal((128,)) * 0.05).astype(np.float32),
        "f2_w": (rng.standard_normal((10, 128)) * 0.05).astype(np.float32),
        "f2_b": (rng.standard_normal((10,)) * 0.05).astype(np.float32),
        "x":    rng.standard_normal((2, 3, 32, 32)).astype(np.float32),  # NCHW
    }


def pytorch_reference(inputs):
    c1 = nn.Conv2d(3, 32, 3, padding=1); c2 = nn.Conv2d(32, 64, 3, padding=1)
    f1 = nn.Linear(64*16*16, 128); f2 = nn.Linear(128, 10)
    pool = nn.MaxPool2d(2, 2)
    with torch.no_grad():
        c1.weight.copy_(torch.from_numpy(inputs["c1_w"])); c1.bias.copy_(torch.from_numpy(inputs["c1_b"]))
        c2.weight.copy_(torch.from_numpy(inputs["c2_w"])); c2.bias.copy_(torch.from_numpy(inputs["c2_b"]))
        f1.weight.copy_(torch.from_numpy(inputs["f1_w"])); f1.bias.copy_(torch.from_numpy(inputs["f1_b"]))
        f2.weight.copy_(torch.from_numpy(inputs["f2_w"])); f2.bias.copy_(torch.from_numpy(inputs["f2_b"]))
    h = torch.relu(c1(torch.from_numpy(inputs["x"])))
    h = pool(torch.relu(c2(h)))
    h = h.view(h.size(0), -1)
    h = torch.relu(f1(h))
    return {"logits": f2(h).detach().numpy()}


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("m3: fixtures written")
