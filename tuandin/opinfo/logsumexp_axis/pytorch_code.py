"""Tests `torch.logsumexp(x, dim=, keepdim=)` — numerically stable log-sum-exp.

JAX equivalent is `jax.scipy.special.logsumexp`. A naive translation
(`jnp.log(jnp.sum(jnp.exp(x), axis=K))`) loses precision for large values.
"""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    # Mix small and large magnitudes to stress numerical stability.
    return {"x": (rng.standard_normal((4, 7)) * 30).astype(np.float32)}


def compute(inputs):
    x = torch.from_numpy(inputs["x"])
    return {
        "lse_dim1": torch.logsumexp(x, dim=1).numpy(),
        "lse_dim0_keepdim": torch.logsumexp(x, dim=0, keepdim=True).numpy(),
    }
