"""JAX translation of e7: train -> save -> load -> predict.

Faithful to PyTorch:
- Same SimpleModel = nn.Linear(1, 1), default init.
- 100 epochs SGD lr=0.01 on X = rand(100,1), y = 3X + 2 + 0.1*randn.
- Save state to disk, load it back into a fresh model, run inference.

PyTorch uses torch.save (pickle of state_dict). JAX has no canonical "state_dict"
format; we use pickle of the params pytree, which is the closest 1:1 analogue.
For production, prefer orbax-checkpoint; pickle is fine for this didactic case.

Speed notes: trivial workload; JAX ≈ PyTorch.
"""
import pickle
import jax
import jax.numpy as jnp
import optax


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
    key, subkey = jax.random.split(key)
    params = init_linear(subkey, 1, 1)

    key, kx, ky = jax.random.split(key, 3)
    X = jax.random.uniform(kx, (100, 1))
    y = 3 * X + 2 + jax.random.normal(ky, (100, 1)) * 0.1

    opt = optax.sgd(0.01)
    opt_state = opt.init(params)

    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for _ in range(100):
        params, opt_state, _ = train_step(params, opt_state, X, y)

    # Save params (analogue of torch.save(model.state_dict(), "model.pth")).
    with open("model.pkl", "wb") as f:
        pickle.dump(jax.device_get(params), f)

    # Load into a fresh container.
    with open("model.pkl", "rb") as f:
        loaded_params = pickle.load(f)

    X_test = jnp.array([[0.5], [1.0], [1.5]])
    preds = model_apply(loaded_params, X_test)
    print(f"Predictions after loading: {preds}")


if __name__ == "__main__":
    main()
