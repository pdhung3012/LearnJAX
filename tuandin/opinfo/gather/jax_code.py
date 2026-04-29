"""JAX: torch.gather(x, dim=1, index=idx) -> jnp.take_along_axis(x, idx, axis=1)."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    x = jnp.asarray(inputs["x"])
    index = jnp.asarray(inputs["index"])
    out = jnp.take_along_axis(x, index, axis=1)
    return {"out": np.asarray(out)}
