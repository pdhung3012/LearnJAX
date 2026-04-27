"""JAX translation of e2: linear regression with mini-batch DataLoader from CSV.

Faithful to PyTorch:
- Generates the same data, writes it to data.csv, reads it back via pandas.
- batch_size=32, shuffle=True per epoch.
- 1000 epochs; loss printed every 100 epochs is the *last batch* loss
  (matches PyTorch's 'loss' variable after the inner loop).

Speed notes:
- We replace torch DataLoader with jnp.permutation + array slicing; DataLoader is
  one of the slowest things in PyTorch for tiny tensors (per-batch Python overhead,
  pinning, collation). JAX should be markedly faster here.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd


# ---- Contract API used by test_equivalence.py ------------------------------
def compute(inputs):
    """Run e2's deterministic core: Linear(1,1) forward with caller-supplied weights.

    Args:
      inputs: dict with keys "W" (out, in)=(1, 1), "b" (out,)=(1,), "X" (N, 1).
    Returns:
      dict with key "predictions" of shape (N, 1).
    """
    W = jnp.asarray(inputs["W"]).T
    b = jnp.asarray(inputs["b"])
    X = jnp.asarray(inputs["X"])
    return {"predictions": np.asarray(X @ W + b)}


def init_linear(key, in_features, out_features):
    bound = 1.0 / jnp.sqrt(in_features)
    kw, kb = jax.random.split(key)
    W = jax.random.uniform(kw, (in_features, out_features), minval=-bound, maxval=bound)
    b = jax.random.uniform(kb, (out_features,), minval=-bound, maxval=bound)
    return {"W": W, "b": b}


def model_apply(params, x):
    return x @ params["W"] + params["b"]


def loss_fn(params, x, y):
    pred = model_apply(params, x)
    return jnp.mean((pred - y) ** 2)


def main():
    # Generate synthetic data and persist to CSV (mirrors PyTorch script).
    key = jax.random.PRNGKey(42)
    key, kx, ky = jax.random.split(key, 3)
    X = jax.random.uniform(kx, (100, 1)) * 10.0
    y = 2 * X + 3 + jax.random.normal(ky, (100, 1))
    data = jnp.concatenate([X, y], axis=1)
    pd.DataFrame(jax.device_get(data), columns=["X", "y"]).to_csv("data.csv", index=False)

    # Load back from CSV (mirrors LinearRegressionDataset).
    df = pd.read_csv("data.csv")
    X = jnp.asarray(df["X"].values, dtype=jnp.float32).reshape(-1, 1)
    y = jnp.asarray(df["y"].values, dtype=jnp.float32).reshape(-1, 1)

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

    n = X.shape[0]
    batch_size = 32
    epochs = 1000
    for epoch in range(epochs):
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n)
        Xs = X[perm]
        ys = y[perm]
        loss = jnp.array(0.0)
        for start in range(0, n, batch_size):
            xb = Xs[start:start + batch_size]
            yb = ys[start:start + batch_size]
            params, opt_state, loss = train_step(params, opt_state, xb, yb)
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
