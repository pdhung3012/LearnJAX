"""Tests outer-product-via-broadcasting: a[None, :] + b[:, None] -> outer add."""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "a": rng.standard_normal((6,)).astype(np.float32),
        "b": rng.standard_normal((4,)).astype(np.float32),
    }


def compute(inputs):
    a = torch.from_numpy(inputs["a"])
    b = torch.from_numpy(inputs["b"])
    out = a[None, :] + b[:, None]    # shape (4, 6)
    return {"out": out.numpy()}
