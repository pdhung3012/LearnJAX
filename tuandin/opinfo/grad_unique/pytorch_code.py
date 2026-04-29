"""Tests gradient through `torch.unique` — both PT and JAX surface this as
non-differentiable (returns zero gradient or errors). We assert the
non-erroring path: gradient w.r.t. a function of unique() that's
differentiable through other branches.

Specifically: out = sin(x).sum() — unique is computed but not used in the
loss path. Both PT and JAX should produce gradient = cos(x).
"""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    return {"x": rng.standard_normal((10,)).astype(np.float32)}


def compute(inputs):
    x = torch.tensor(inputs["x"], requires_grad=True)
    _ = torch.unique(x.detach())     # called for side effect; .detach() so no grad path
    loss = torch.sin(x).sum()
    loss.backward()
    return {"grad": x.grad.numpy()}
