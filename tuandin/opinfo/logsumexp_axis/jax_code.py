"""JAX translation: torch.logsumexp -> jax.scipy.special.logsumexp."""
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp


def compute(inputs):
    x = jnp.asarray(inputs["x"])
    return {
        "lse_dim1": np.asarray(logsumexp(x, axis=1)),
        "lse_dim0_keepdim": np.asarray(logsumexp(x, axis=0, keepdims=True)),
    }
