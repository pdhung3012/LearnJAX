"""Tests `torch.argmin(x, dim=, keepdim=)`."""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    return {"x": rng.standard_normal((4, 7, 5)).astype(np.float32)}


def compute(inputs):
    x = torch.from_numpy(inputs["x"])
    return {
        "argmin_dim0": torch.argmin(x, dim=0).numpy().astype(np.int64),
        "argmin_dim2_keepdim": torch.argmin(x, dim=2, keepdim=True).numpy().astype(np.int64),
    }
