"""JAX translation of e1: simple linear regression with full-batch SGD.

Faithful to PyTorch:
- nn.Linear(1, 1) default init: uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) for W and b.
- nn.MSELoss reduction='mean'.
- optim.SGD lr=0.01.
- 1000 full-batch epochs, log every 100.

Speed notes:
- jit'd train_step keeps the whole forward+backward+update on a single XLA call.
- For full-batch on a 100x1 problem the per-step compute is tiny; XLA dispatch
  overhead vs PyTorch's eager dispatch ends up similar on CPU. JAX is typically
  competitive-to-faster here once the jit cache is warm (compile cost is paid once).

Note on RNG: torch.manual_seed(42) is not reproducible bit-for-bit in JAX; we use
PRNGKey(42) which yields the same statistical distribution but different samples.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax


def init_linear(key, in_features, out_features):
    bound = 1.0 / jnp.sqrt(in_features)
    kw, kb = jax.random.split(key)
    W = jax.random.uniform(kw, (in_features, out_features), minval=-bound, maxval=bound)
    b = jax.random.uniform(kb, (out_features,), minval=-bound, maxval=bound)
    return {"W": W, "b": b}


def model_apply(params, x):
    return x @ params["W"] + params["b"]


# ---- Contract API used by test_equivalence.py and the eval harness ----------
#
# `compute(inputs)` is the single entry point a translation must implement.
# It takes a dict of numpy arrays (matching the keys in inputs.npz) and returns
# a dict of numpy arrays (matching the keys in expected.npz). Internal JAX
# representation/jit choices are hidden from callers.
def compute(inputs):
    """Run e1's deterministic core: nn.Linear(1, 1) forward pass.

    Args:
      inputs: dict with keys "W" (out, in)=(1, 1), "b" (out,)=(1,), "X" (N, 1).
    Returns:
      dict with key "predictions" of shape (N, 1).
    """
    # PyTorch nn.Linear weight is (out, in); JAX kernel convention is (in, out).
    W = jnp.asarray(inputs["W"]).T
    b = jnp.asarray(inputs["b"])
    X = jnp.asarray(inputs["X"])
    return {"predictions": np.asarray(X @ W + b)}


def loss_fn(params, x, y):
    pred = model_apply(params, x)
    return jnp.mean((pred - y) ** 2)


def main():
    key = jax.random.PRNGKey(42)
    key, kx, ky = jax.random.split(key, 3)
    X = jax.random.uniform(kx, (100, 1)) * 10.0
    y = 2 * X + 3 + jax.random.normal(ky, (100, 1))

    key, subkey = jax.random.split(key)
    params = init_linear(subkey, 1, 1)

    opt = optax.sgd(0.01)
    opt_state = opt.init(params)

    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    epochs = 1000
    for epoch in range(epochs):
        params, opt_state, loss = train_step(params, opt_state, X, y)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

    w = float(params["W"][0, 0])
    b = float(params["b"][0])
    print(f"Learned weight: {w:.4f}, Learned bias: {b:.4f}")

    X_test = jnp.array([[4.0], [7.0]])
    preds = model_apply(params, X_test)
    print(f"Predictions for {X_test.tolist()}: {preds.tolist()}")


if __name__ == "__main__":
    main()
