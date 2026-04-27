"""Freeze fixtures for e4: Huber loss formula (means over the loss tensor)."""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "y_pred":  (rng.standard_normal((100, 1)) * 3).astype(np.float32),
        "y_true":  (rng.standard_normal((100, 1)) * 3).astype(np.float32),
        "delta":   np.array(1.0, dtype=np.float32),
    }


def pytorch_reference(inputs):
    pred = torch.from_numpy(inputs["y_pred"])
    true = torch.from_numpy(inputs["y_true"])
    delta = float(inputs["delta"])
    err = torch.abs(pred - true)
    loss = torch.where(err <= delta,
                       0.5 * err ** 2,
                       delta * (err - 0.5 * delta)).mean()
    return {"loss": loss.numpy()}


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("e4: fixtures written")
