"""JAX translation of e6: linear regression with TensorBoard logging.

Faithful to PyTorch:
- y = 3*X + 5 + noise (note: different from e1 which uses 2*X + 3).
- Logs every-epoch loss to TensorBoard via torch.utils.tensorboard.SummaryWriter
  (the same writer is used here so the resulting events file is identical in format).
- 100 epochs, console print every 10 epochs.

Speed notes:
- Per-step the loss must be brought to host (writer.add_scalar wants a Python float).
  This forces a device->host sync each step in both implementations, so the
  TensorBoard write itself is the dominant cost — JAX is roughly on par here.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from torch.utils.tensorboard import SummaryWriter


# ---- Contract API used by test_equivalence.py ------------------------------
def compute(inputs):
    """Linear(1,1) forward with caller-supplied weights."""
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
    return jnp.mean((model_apply(params, x) - y) ** 2)


def main():
    key = jax.random.PRNGKey(42)
    key, kx, ky = jax.random.split(key, 3)
    X = jax.random.uniform(kx, (100, 1)) * 10.0
    y = 3 * X + 5 + jax.random.normal(ky, (100, 1))

    writer = SummaryWriter(log_dir="runs/linear_regression")

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

    epochs = 100
    for epoch in range(epochs):
        params, opt_state, loss = train_step(params, opt_state, X, y)
        writer.add_scalar("Loss/train", float(loss), epoch)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {float(loss):.4f}")

    writer.close()


if __name__ == "__main__":
    main()
