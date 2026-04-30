"""JAX: jnp.exp matches PT semantics (overflow to inf, underflow to 0)."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    x = jnp.asarray(inputs["x"])
    return {"out": np.asarray(jnp.exp(x))}
