"""JAX: gradient through jnp.where with finite-safe branches."""
import jax
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    thresh = float(inputs["thresh"])

    def loss_fn(x):
        safe = jnp.where(x > thresh, x ** 2, x)
        return jnp.sum(safe)

    grad = jax.grad(loss_fn)(jnp.asarray(inputs["x"]))
    return {"grad": np.asarray(grad)}
