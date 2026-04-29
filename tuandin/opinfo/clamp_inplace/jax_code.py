"""JAX translation: x.clamp_(min, max) -> x = jnp.clip(x, min, max)."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    x = jnp.asarray(inputs["x"])
    x = jnp.clip(x, float(inputs["lo"]), float(inputs["hi"]))
    out = x ** 2
    return {"out": np.asarray(out)}
