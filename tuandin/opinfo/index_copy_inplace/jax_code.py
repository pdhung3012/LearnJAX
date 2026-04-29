"""JAX translation: x.index_copy_(0, idx, src) -> x = x.at[idx].set(src)."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    x = jnp.asarray(inputs["x"])
    src = jnp.asarray(inputs["src"])
    index = jnp.asarray(inputs["index"])
    x = x.at[index].set(src)
    out = jnp.sum(x, axis=1)
    return {"out": np.asarray(out)}
