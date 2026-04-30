"""JAX translation: torch.cummax values -> jnp.maximum.accumulate."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    x = jnp.asarray(inputs["x"])
    values = jnp.maximum.accumulate(x, axis=1)
    return {"cummax_values_dim1": np.asarray(values)}
