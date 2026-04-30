"""JAX: jnp.where with broadcast cond+a+scalar."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    cond = jnp.asarray(inputs["cond"])
    a = jnp.asarray(inputs["a"])
    b_val = float(inputs["b"])
    out = jnp.where(cond, a, b_val)
    return {"out": np.asarray(out)}
