"""JAX: bool * float promotes to float."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    b = jnp.asarray(inputs["b"])
    f = jnp.asarray(inputs["f"])
    out = b * f
    return {"out": np.asarray(out)}
