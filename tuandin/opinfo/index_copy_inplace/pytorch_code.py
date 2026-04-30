"""Tests `Tensor.index_copy_(dim, index, src)` — copy `src` into `x` at the given indices.

JAX equivalent uses the functional `.at[idx].set(src)` syntax.
"""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "x":     rng.standard_normal((6, 4)).astype(np.float32),
        "src":   rng.standard_normal((3, 4)).astype(np.float32),
        "index": np.array([0, 2, 4], dtype=np.int64),
    }


def compute(inputs):
    x = torch.from_numpy(inputs["x"]).clone()
    src = torch.from_numpy(inputs["src"])
    index = torch.from_numpy(inputs["index"])
    x.index_copy_(0, index, src)     # rows 0, 2, 4 of x are replaced with src.
    out = x.sum(dim=1)
    return {"out": out.numpy()}
