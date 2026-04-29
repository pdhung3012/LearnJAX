"""JAX: x[mask] works the same outside jit. We use it directly here."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    x = jnp.asarray(inputs["x"])
    mask = jnp.asarray(inputs["mask"])
    out = x[mask]
    return {"out": np.asarray(out)}
