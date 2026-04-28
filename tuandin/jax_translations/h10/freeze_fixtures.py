"""Freeze fixtures for h10: Grad-CAM combine formula.

Inputs are the *intermediate* activations and gradients at the chosen conv
layer (we don't run the full ResNet18 — that requires downloading pretrained
weights).
"""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    # PyTorch convention: (B, C, H, W).
    return {
        "activations": rng.standard_normal((1, 32, 7, 7)).astype(np.float32),
        "gradients":  (rng.standard_normal((1, 32, 7, 7)) * 0.1).astype(np.float32),
    }


def pytorch_reference(inputs):
    a = torch.from_numpy(inputs["activations"])
    g = torch.from_numpy(inputs["gradients"])
    weights = g.mean(dim=[2, 3], keepdim=True)
    h = (weights * a).sum(dim=1).squeeze().relu()
    h = h / (h.max() + 1e-8)
    return {"heatmap": h.numpy()}


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("h10: fixtures written")
