"""Tests gradient through `torch.sort`. The gradient of sort w.r.t. its
input is well-defined (a permutation matrix); ties are broken arbitrarily
but for distinct inputs both PT and JAX should agree.
"""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    # Distinct values to avoid tie-breaking divergence.
    return {"x": (rng.standard_normal((4, 6)) + np.arange(24).reshape(4, 6) * 0.001).astype(np.float32)}


def compute(inputs):
    x = torch.tensor(inputs["x"], requires_grad=True)
    sorted_vals, _idx = torch.sort(x, dim=1)
    loss = (sorted_vals ** 2).sum()
    loss.backward()
    return {"grad": x.grad.numpy()}
