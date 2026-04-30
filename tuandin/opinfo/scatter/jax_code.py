"""JAX: torch.scatter(x, dim=1, index, src) -> use jnp.put_along_axis or .at[...].set(...)."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    x = jnp.asarray(inputs["x"])
    index = jnp.asarray(inputs["index"])
    src = jnp.asarray(inputs["src"])
    # Build (row_idx, col_idx) pairs for advanced indexing.
    rows = jnp.arange(x.shape[0])[:, None]     # (4, 1)
    out = x.at[rows, index].set(src)
    return {"out": np.asarray(out)}
