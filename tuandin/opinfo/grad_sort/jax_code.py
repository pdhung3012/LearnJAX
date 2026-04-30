"""JAX: gradient through jnp.sort via jax.grad."""
import jax
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    def loss_fn(x):
        sorted_vals = jnp.sort(x, axis=1)
        return jnp.sum(sorted_vals ** 2)

    grad = jax.grad(loss_fn)(jnp.asarray(inputs["x"]))
    return {"grad": np.asarray(grad)}
