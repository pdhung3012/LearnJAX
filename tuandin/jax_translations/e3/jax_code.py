"""JAX translation of e3: linear regression with custom activation tanh(x)+x.

Faithful to PyTorch:
- nn.Linear(1, 1) default init.
- Custom activation: tanh(x) + x (residual tanh).
- MSE, SGD lr=0.01, 1000 epochs, log every 100.
- Final scatter+line plot.

Speed notes:
- Same shape as e1 with one extra elementwise op; jit'd train_step erases the
  Python overhead, so JAX should be at-or-faster than PyTorch here.
"""
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt


def init_linear(key, in_features, out_features):
    bound = 1.0 / jnp.sqrt(in_features)
    kw, kb = jax.random.split(key)
    W = jax.random.uniform(kw, (in_features, out_features), minval=-bound, maxval=bound)
    b = jax.random.uniform(kb, (out_features,), minval=-bound, maxval=bound)
    return {"W": W, "b": b}


def custom_activation(x):
    return jnp.tanh(x) + x


def model_apply(params, x):
    return custom_activation(x @ params["W"] + params["b"])


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

    plt.figure(figsize=(4, 4))
    Xn = jax.device_get(X)
    yn = jax.device_get(y)
    plt.scatter(Xn, yn, label="Training Data")
    plt.plot(Xn, w * Xn + b, "r", label="Model Fit")
    plt.legend()
    plt.show()

    X_test = jnp.array([[4.0], [7.0]])
    preds = model_apply(params, X_test)
    print(f"Predictions for {X_test.tolist()}: {preds.tolist()}")


if __name__ == "__main__":
    main()
