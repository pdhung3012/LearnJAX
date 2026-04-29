"""Tests `torch.index_select(x, dim, index)` — pick rows/cols/etc. along an axis."""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "x":     rng.standard_normal((6, 4, 5)).astype(np.float32),
        "idx_d0": np.array([0, 2, 5, 1], dtype=np.int64),
        "idx_d1": np.array([3, 0], dtype=np.int64),
    }


def compute(inputs):
    x = torch.from_numpy(inputs["x"])
    return {
        "out_d0": torch.index_select(x, 0, torch.from_numpy(inputs["idx_d0"])).numpy(),
        "out_d1": torch.index_select(x, 1, torch.from_numpy(inputs["idx_d1"])).numpy(),
    }
