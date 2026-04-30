"""JAX: torch.index_select(x, dim, idx) -> jnp.take(x, idx, axis=dim)."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    x = jnp.asarray(inputs["x"])
    return {
        "out_d0": np.asarray(jnp.take(x, jnp.asarray(inputs["idx_d0"]), axis=0)),
        "out_d1": np.asarray(jnp.take(x, jnp.asarray(inputs["idx_d1"]), axis=1)),
    }
