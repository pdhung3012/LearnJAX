"""JAX: gradient through jax.lax.top_k via jax.grad."""
import jax
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    def loss_fn(x):
        top_vals, _idx = jax.lax.top_k(x, k=3)    # JAX top_k operates on last axis only.
        return jnp.sum(top_vals ** 2)

    grad = jax.grad(loss_fn)(jnp.asarray(inputs["x"]))
    return {"grad": np.asarray(grad)}
