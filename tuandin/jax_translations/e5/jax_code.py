"""JAX translation of e5: 2->10->1 MLP with ReLU and Adam.

Faithful to PyTorch:
- nn.Linear(2,10), ReLU, nn.Linear(10,1) with PyTorch default init for both.
- nn.MSELoss, Adam lr=0.01, 1000 epochs, full batch, log every 100.

Speed notes: Adam state + MLP fits well in jit; JAX should be at-or-faster
than PyTorch for this size on CPU.
"""
import jax
import jax.numpy as jnp
import optax


def init_linear(key, in_features, out_features):
    bound = 1.0 / jnp.sqrt(in_features)
    kw, kb = jax.random.split(key)
    W = jax.random.uniform(kw, (in_features, out_features), minval=-bound, maxval=bound)
    b = jax.random.uniform(kb, (out_features,), minval=-bound, maxval=bound)
    return {"W": W, "b": b}


def init_mlp(key):
    k1, k2 = jax.random.split(key)
    return {"l1": init_linear(k1, 2, 10), "l2": init_linear(k2, 10, 1)}


def model_apply(params, x):
    h = jax.nn.relu(x @ params["l1"]["W"] + params["l1"]["b"])
    return h @ params["l2"]["W"] + params["l2"]["b"]


def loss_fn(params, x, y):
    return jnp.mean((model_apply(params, x) - y) ** 2)


def main():
    key = jax.random.PRNGKey(42)
    key, kx, ky = jax.random.split(key, 3)
    X = jax.random.uniform(kx, (100, 2)) * 10.0
    # PyTorch: y = (X[:,0] + X[:,1]*2).unsqueeze(1) + randn(100,1)
    y = (X[:, 0] + X[:, 1] * 2).reshape(-1, 1) + jax.random.normal(ky, (100, 1))

    key, subkey = jax.random.split(key)
    params = init_mlp(subkey)
    opt = optax.adam(0.01)
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

    X_test = jnp.array([[4.0, 3.0], [7.0, 8.0]])
    preds = model_apply(params, X_test)
    print(f"Predictions for {X_test.tolist()}: {preds.tolist()}")


if __name__ == "__main__":
    main()
