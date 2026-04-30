"""JAX: jnp.abs on complex64 returns float32 magnitude (same as PT)."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    z = jnp.asarray(inputs["z"])
    out = jnp.abs(z)
    return {"out": np.asarray(out)}
