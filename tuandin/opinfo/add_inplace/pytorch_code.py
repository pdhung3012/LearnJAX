"""Tests `Tensor.add_(y)` — in-place addition.

JAX has no in-place ops. Translating `x.add_(y)` requires reassignment:
`x = x + y`. A common cheap-LLM mistake is `jnp.add(x, y)` without
reassignment — `x` stays unchanged and downstream uses of `x` are wrong.

This case uses `x` AFTER the in-place mutation, so the bug surfaces.
"""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "x": rng.standard_normal((4, 8)).astype(np.float32),
        "y": rng.standard_normal((4, 8)).astype(np.float32),
        "z": rng.standard_normal((4, 8)).astype(np.float32),
    }


def compute(inputs):
    x = torch.from_numpy(inputs["x"]).clone()
    y = torch.from_numpy(inputs["y"])
    z = torch.from_numpy(inputs["z"])
    x.add_(y)            # mutate x in-place
    out = x * z          # downstream use of x — this is what catches the bug
    return {"out": out.numpy()}
