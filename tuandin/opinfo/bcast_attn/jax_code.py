"""JAX: same broadcasting semantics as PT for elementwise multiply."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    q = jnp.asarray(inputs["q"])
    k = jnp.asarray(inputs["k"])
    out = q * k
    return {"out": np.asarray(out)}
