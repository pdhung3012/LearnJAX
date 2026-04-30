"""Tests `torch.argmax(x, dim=)` — index of max along an axis.

JAX `jnp.argmax(x, axis=)` matches signature, but the dim/axis argument
naming differs and the int dtype is platform-dependent (PT returns int64;
JAX returns int32 on CPU by default).
"""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    return {"x": rng.standard_normal((4, 7, 5)).astype(np.float32)}


def compute(inputs):
    x = torch.from_numpy(inputs["x"])
    return {
        "argmax_dim1": torch.argmax(x, dim=1).numpy().astype(np.int64),
        "argmax_dim_neg1": torch.argmax(x, dim=-1).numpy().astype(np.int64),
        "argmax_flat": torch.argmax(x).numpy().astype(np.int64).reshape(()),
    }
