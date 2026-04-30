"""JAX: jax.grad of sin(x).sum() — unique is on a non-grad branch."""
import jax
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    def loss_fn(x):
        return jnp.sum(jnp.sin(x))

    grad = jax.grad(loss_fn)(jnp.asarray(inputs["x"]))
    return {"grad": np.asarray(grad)}
