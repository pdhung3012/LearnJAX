"""Tests `torch.cummax(x, dim=)` — running max along an axis. Returns (values, indices).

JAX has `jnp.maximum.accumulate(x, axis=)` for the values; for the indices
the cheap LLM has to combine `jnp.argmax`-style reasoning with running
state. We test only the values here (indices ARE platform-dependent on ties).
"""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    return {"x": rng.standard_normal((4, 7)).astype(np.float32)}


def compute(inputs):
    x = torch.from_numpy(inputs["x"])
    values, _indices = torch.cummax(x, dim=1)
    return {"cummax_values_dim1": values.numpy()}
