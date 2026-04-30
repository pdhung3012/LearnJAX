"""JAX translation: torch.argmax(x, dim=K) -> jnp.argmax(x, axis=K)."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    x = jnp.asarray(inputs["x"])
    return {
        "argmax_dim1": np.asarray(jnp.argmax(x, axis=1)).astype(np.int64),
        "argmax_dim_neg1": np.asarray(jnp.argmax(x, axis=-1)).astype(np.int64),
        "argmax_flat": np.asarray(jnp.argmax(x)).astype(np.int64).reshape(()),
    }
