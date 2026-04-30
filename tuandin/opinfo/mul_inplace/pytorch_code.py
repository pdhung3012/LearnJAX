"""Tests `Tensor.mul_(y)` — in-place multiplication. See add_inplace for rationale."""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "x": rng.standard_normal((4, 8)).astype(np.float32),
        "y": rng.standard_normal((4, 8)).astype(np.float32),
        "bias": rng.standard_normal((8,)).astype(np.float32),
    }


def compute(inputs):
    x = torch.from_numpy(inputs["x"]).clone()
    y = torch.from_numpy(inputs["y"])
    bias = torch.from_numpy(inputs["bias"])
    x.mul_(y)
    out = x + bias
    return {"out": out.numpy()}
