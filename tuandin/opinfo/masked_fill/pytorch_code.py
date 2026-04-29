"""Tests `Tensor.masked_fill(mask, value)` — replace entries where mask is True."""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "x":    rng.standard_normal((4, 8)).astype(np.float32),
        "mask": (rng.standard_normal((4, 8)) > 0.5),
        "value": np.array(-1e9, dtype=np.float32),
    }


def compute(inputs):
    x = torch.from_numpy(inputs["x"]).clone()
    mask = torch.from_numpy(inputs["mask"])
    out = x.masked_fill(mask, float(inputs["value"]))
    return {"out": out.numpy()}
