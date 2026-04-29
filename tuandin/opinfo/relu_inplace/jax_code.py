"""JAX translation: F.relu_(x) -> x = jax.nn.relu(x). JAX has no in-place variant."""
import jax
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    x = jnp.asarray(inputs["x"])
    scale = jnp.asarray(inputs["scale"])
    x = jax.nn.relu(x)
    out = x * scale
    return {"out": np.asarray(out)}
