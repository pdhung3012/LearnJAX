"""JAX translation: x.add_(y) -> x = x + y (functional reassignment)."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    x = jnp.asarray(inputs["x"])
    y = jnp.asarray(inputs["y"])
    z = jnp.asarray(inputs["z"])
    x = x + y           # JAX functional equivalent of x.add_(y)
    out = x * z
    return {"out": np.asarray(out)}
