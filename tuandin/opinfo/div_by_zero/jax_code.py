"""JAX: same float32 division-by-zero semantics as PT (inf/-inf, 0/0=nan)."""
import jax.numpy as jnp
import numpy as np


def compute(inputs):
    n = jnp.asarray(inputs["num"])
    d = jnp.asarray(inputs["denom"])
    out = n / d
    out = jnp.nan_to_num(out, nan=-99.0)
    return {"out": np.asarray(out)}
