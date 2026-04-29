"""Tests `torch.where(cond, a, b)` with partial broadcasting:
cond shape (B, 1, S), a shape (B, H, S), b is a scalar.
"""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    cond = (rng.standard_normal((2, 1, 5)) > 0)
    return {
        "cond": cond,
        "a":    rng.standard_normal((2, 4, 5)).astype(np.float32),
        "b":    np.array(-1.0, dtype=np.float32),
    }


def compute(inputs):
    cond = torch.from_numpy(inputs["cond"])
    a = torch.from_numpy(inputs["a"])
    b_val = float(inputs["b"])
    out = torch.where(cond, a, b_val)
    return {"out": out.numpy()}
