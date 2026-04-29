"""Tests division-by-zero behavior: PT and JAX both produce inf/-inf for x/0
and nan for 0/0 in float32. We confirm this match.
"""
import numpy as np
import torch


def make_inputs():
    return {
        "num":   np.array([1.0, -1.0, 0.0, 2.0], dtype=np.float32),
        "denom": np.array([0.0,  0.0, 0.0, 4.0], dtype=np.float32),
    }


def compute(inputs):
    n = torch.from_numpy(inputs["num"])
    d = torch.from_numpy(inputs["denom"])
    out = n / d
    # Replace NaN (from 0/0) with a sentinel so the equivalence check is
    # well-defined (np.allclose treats NaN as unequal by default).
    out = torch.nan_to_num(out, nan=-99.0)
    return {"out": out.numpy()}
