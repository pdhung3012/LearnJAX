"""JAX: int * float promotes to float32. We just compute and return."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    i = jnp.asarray(inputs["i"])
    f = jnp.asarray(inputs["f"])
    out = i * f
    return {"out": np.asarray(out)}
