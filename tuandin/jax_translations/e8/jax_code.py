"""JAX translation of e8: RMSNorm.

Faithful to PyTorch:
- Learnable per-feature `scale` parameter (gamma), initialized to ones.
- Forward: x / sqrt(mean(x^2, axis=-1, keepdims=True) + eps) * scale.

Notes:
- Uses Flax `nn.Module` to mirror PyTorch's `nn.Module` API. The PyTorch
  default `eps=1e-8` is preserved; production LLMs typically use 1e-5 / 1e-6.

Speed: trivial elementwise; jit'd JAX is essentially identical to PyTorch eager
on CPU for this workload (microseconds).
"""
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn


# ---- Contract API used by test_equivalence.py ------------------------------
def compute(inputs):
    """RMSNorm forward with caller-supplied scale.

    Args:
      inputs: dict with "x" (..., dim), "scale" (dim,), "eps" (0-d float).
    Returns:
      dict with "output" same shape as x.
    """
    x = jnp.asarray(inputs["x"])
    scale = jnp.asarray(inputs["scale"])
    eps = float(inputs["eps"])
    norm = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return {"output": np.asarray((x / norm) * scale)}


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-8

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.dim,))
        norm = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / norm) * scale


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (3, 5))
    model = RMSNorm(dim=5)
    params = model.init(key, x)
    out = model.apply(params, x)
    print("input :", x)
    print("output:", out)
    print("shape :", out.shape)
    assert out.shape == (3, 5), "Output shape mismatch"
