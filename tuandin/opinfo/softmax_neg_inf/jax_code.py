"""JAX: jax.nn.softmax with NaN handling matched to PT."""
import jax
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    x = jnp.asarray(inputs["x"])
    out = jax.nn.softmax(x, axis=-1)
    out = jnp.nan_to_num(out, nan=-99.0)
    return {"out": np.asarray(out)}
