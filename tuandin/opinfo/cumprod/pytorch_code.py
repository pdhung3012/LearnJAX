"""Tests `torch.cumprod(x, dim=)` — running product along an axis."""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    # Keep magnitudes near 1 so the running product doesn't blow up.
    return {"x": (1.0 + rng.standard_normal((3, 6)) * 0.1).astype(np.float32)}


def compute(inputs):
    x = torch.from_numpy(inputs["x"])
    return {
        "cumprod_dim0": torch.cumprod(x, dim=0).numpy(),
        "cumprod_dim1": torch.cumprod(x, dim=1).numpy(),
    }
