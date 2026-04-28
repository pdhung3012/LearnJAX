"""Freeze fixtures for h5: cross-entropy loss (the per-step loss in seq2seq teacher forcing)."""
import numpy as np
import torch
import torch.nn.functional as F


def make_inputs():
    rng = np.random.default_rng(0)
    B, V = 8, 20
    return {
        "logits": (rng.standard_normal((B, V)) * 2).astype(np.float32),
        "labels": rng.integers(0, V, (B,)).astype(np.int64),
    }


def pytorch_reference(inputs):
    logits = torch.from_numpy(inputs["logits"])
    labels = torch.from_numpy(inputs["labels"])
    return {"cross_entropy": F.cross_entropy(logits, labels).numpy()}


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("h5: fixtures written")
