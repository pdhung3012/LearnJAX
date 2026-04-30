"""JAX: same broadcasting via None-indexing as PT."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    a = jnp.asarray(inputs["a"])
    b = jnp.asarray(inputs["b"])
    out = a[None, :] + b[:, None]
    return {"out": np.asarray(out)}
