"""JAX: torch.scatter_add -> .at[rows, index].add(src) (note: .add, not .set)."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    x = jnp.asarray(inputs["x"])
    index = jnp.asarray(inputs["index"])
    src = jnp.asarray(inputs["src"])
    rows = jnp.arange(x.shape[0])[:, None]
    out = x.at[rows, index].add(src)
    return {"out": np.asarray(out)}
