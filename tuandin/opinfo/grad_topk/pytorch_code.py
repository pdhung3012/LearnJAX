"""Tests gradient through `torch.topk` — the gradient flows only to the
elements that were selected; non-selected entries get zero gradient.
"""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    return {"x": (rng.standard_normal((4, 8)) + np.arange(32).reshape(4, 8) * 0.001).astype(np.float32)}


def compute(inputs):
    x = torch.tensor(inputs["x"], requires_grad=True)
    top_vals, _idx = torch.topk(x, k=3, dim=1)
    loss = (top_vals ** 2).sum()
    loss.backward()
    return {"grad": x.grad.numpy()}
