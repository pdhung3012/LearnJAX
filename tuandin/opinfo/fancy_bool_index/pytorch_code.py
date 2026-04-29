"""Tests `x[mask]` boolean fancy indexing — extract entries where mask is True.

Tricky: PyTorch's `x[mask]` returns a 1-D tensor. JAX's `x[mask]` does
the same — but ONLY under jit if the static shape can be inferred.
For dynamic shapes use `jnp.compress` or `jnp.extract`.
"""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 5)).astype(np.float32)
    # Use a deterministic mask with a known number of True entries (so
    # fixed-shape extraction works under jit too).
    mask = (np.arange(20).reshape(4, 5) % 3 == 0)        # 7 True entries
    return {"x": x, "mask": mask.astype(np.bool_)}


def compute(inputs):
    x = torch.from_numpy(inputs["x"])
    mask = torch.from_numpy(inputs["mask"])
    out = x[mask]
    return {"out": out.numpy()}
