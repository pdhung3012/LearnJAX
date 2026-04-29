"""JAX: x.masked_fill(mask, val) -> jnp.where(mask, val, x)."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    x = jnp.asarray(inputs["x"])
    mask = jnp.asarray(inputs["mask"])
    out = jnp.where(mask, float(inputs["value"]), x)
    return {"out": np.asarray(out)}
