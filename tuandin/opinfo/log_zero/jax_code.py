"""JAX: jnp.log(0) -> -inf in float32 (matches PT)."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    x = jnp.asarray(inputs["x"])
    return {"out": np.asarray(jnp.log(x))}
