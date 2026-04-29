"""JAX translation: x.mul_(y) -> x = x * y."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    x = jnp.asarray(inputs["x"])
    y = jnp.asarray(inputs["y"])
    bias = jnp.asarray(inputs["bias"])
    x = x * y
    out = x + bias
    return {"out": np.asarray(out)}
