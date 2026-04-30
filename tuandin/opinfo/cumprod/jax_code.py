"""JAX translation: torch.cumprod -> jnp.cumprod."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    x = jnp.asarray(inputs["x"])
    return {
        "cumprod_dim0": np.asarray(jnp.cumprod(x, axis=0)),
        "cumprod_dim1": np.asarray(jnp.cumprod(x, axis=1)),
    }
