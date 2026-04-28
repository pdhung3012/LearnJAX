"""Freeze fixtures for h4: BCE loss formula + LeakyReLU(0.2) + tanh."""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "p":      np.clip(rng.uniform(0.0, 1.0, (32, 1)).astype(np.float32), 0.01, 0.99),
        "t":      rng.integers(0, 2, (32, 1)).astype(np.float32),
        "z":      rng.standard_normal((10,)).astype(np.float32),
        "eps":    np.array(1e-7, dtype=np.float32),
    }


def bce_pt(p, t, eps):
    p = torch.clamp(p, eps, 1 - eps)
    return -(t * torch.log(p) + (1 - t) * torch.log(1 - p)).mean()


def pytorch_reference(inputs):
    p = torch.from_numpy(inputs["p"]); t = torch.from_numpy(inputs["t"])
    z = torch.from_numpy(inputs["z"])
    return {
        "bce":        bce_pt(p, t, float(inputs["eps"])).numpy(),
        "leaky_relu": torch.nn.functional.leaky_relu(z, 0.2).numpy(),
        "tanh":       torch.tanh(z).numpy(),
    }


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("h4: fixtures written")
