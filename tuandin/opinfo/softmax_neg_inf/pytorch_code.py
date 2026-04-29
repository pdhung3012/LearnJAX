"""Tests `softmax` over a row that's entirely -inf. Both PT and JAX
return NaN for such rows (since exp(-inf) = 0 and 0/0 = NaN). The test
verifies both behave the same.
"""
import numpy as np
import torch


def make_inputs():
    return {
        "x": np.array([
            [1.0, 2.0, 3.0],
            [-np.inf, -np.inf, -np.inf],   # all-mask row
            [0.0, -np.inf, 0.0],
        ], dtype=np.float32),
    }


def compute(inputs):
    x = torch.from_numpy(inputs["x"])
    out = torch.softmax(x, dim=-1)
    # Replace NaN with a sentinel so the equivalence check is well-defined.
    out = torch.nan_to_num(out, nan=-99.0)
    return {"out": out.numpy()}
