"""JAX translation: torch.argmin(x, dim=, keepdim=) -> jnp.argmin(x, axis=, keepdims=)."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    x = jnp.asarray(inputs["x"])
    return {
        "argmin_dim0": np.asarray(jnp.argmin(x, axis=0)).astype(np.int64),
        "argmin_dim2_keepdim": np.asarray(jnp.argmin(x, axis=2, keepdims=True)).astype(np.int64),
    }
