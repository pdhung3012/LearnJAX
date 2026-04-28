"""Freeze fixtures for m4: dice loss formula.

The full pipeline (pretrained ResNet18 + 3D conv head) is too heavy to test;
we contract on the dice score formula which is the case's core scoring metric.
"""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "pred":  np.abs(rng.standard_normal((2, 4, 1, 8, 8))).astype(np.float32),
        "label": (rng.standard_normal((2, 4, 1, 8, 8)) > 0).astype(np.float32),
        "eps":   np.array(1e-8, dtype=np.float32),
    }


def pytorch_reference(inputs):
    pred = torch.from_numpy(inputs["pred"])
    label = torch.from_numpy(inputs["label"])
    eps = float(inputs["eps"])
    num = 2 * torch.sum(pred * label)
    den = torch.sum(pred) + torch.sum(label) + eps
    return {"dice": (num / den).numpy()}


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("m4: fixtures written")
