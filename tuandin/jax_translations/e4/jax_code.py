"""JAX translation of e4: linear regression with Huber loss.

Faithful to PyTorch:
- HuberLoss(delta=1.0) = mean over min(0.5*err^2, delta*(err - 0.5*delta)).
- nn.Linear(1, 1) default init, SGD lr=0.01, 1000 epochs, log every 100.

Speed notes: Same as e1; should be on par with or faster than PyTorch.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax


# ---- Contract API used by test_equivalence.py ------------------------------
def compute(inputs):
    """Huber loss with caller-supplied delta.

    Args:
      inputs: dict with "y_pred", "y_true" (same shape), "delta" (0-d float).
    Returns:
      dict with "loss" (scalar).
    """
    pred = jnp.asarray(inputs["y_pred"])
    true = jnp.asarray(inputs["y_true"])
    delta = float(inputs["delta"])
    err = jnp.abs(pred - true)
    loss = jnp.mean(jnp.where(err <= delta,
                              0.5 * err ** 2,
                              delta * (err - 0.5 * delta)))
    return {"loss": np.asarray(loss)}


def init_linear(key, in_features, out_features):
    bound = 1.0 / jnp.sqrt(in_features)
    kw, kb = jax.random.split(key)
    W = jax.random.uniform(kw, (in_features, out_features), minval=-bound, maxval=bound)
    b = jax.random.uniform(kb, (out_features,), minval=-bound, maxval=bound)
    return {"W": W, "b": b}


def model_apply(params, x):
    return x @ params["W"] + params["b"]


def huber_loss(pred, target, delta=1.0):
    error = jnp.abs(pred - target)
    quadratic = 0.5 * error ** 2
    linear = delta * (error - 0.5 * delta)
    return jnp.mean(jnp.where(error <= delta, quadratic, linear))


def loss_fn(params, x, y):
    return huber_loss(model_apply(params, x), y, delta=1.0)


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
