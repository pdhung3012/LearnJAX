# jax_code_fixed.py
import pickle
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import jit, random


def init_params(rng_key: jax.Array) -> Dict[str, jax.Array]:
    """
    PYTORCH EQUIVALENT:
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(1, 1)

    TRANSLATION NOTES:
        - PyTorch nn.Linear(1,1) is represented as a params dict with:
            w: shape (1, 1)
            b: shape (1,)
        - RNG is explicit in JAX.

    MOCK INJECTION:
        - {"w": jnp.array([[1.0]], dtype=jnp.float32),
           "b": jnp.array([0.0], dtype=jnp.float32)}
    """
    k_w, k_b = random.split(rng_key)
    w = random.normal(k_w, (1, 1), dtype=jnp.float32) * 0.01
    b = random.normal(k_b, (1,), dtype=jnp.float32) * 0.01
    return {"w": w, "b": b}


def model_apply(params: Dict[str, jax.Array], x: jax.Array) -> jax.Array:
    """
    PYTORCH EQUIVALENT:
        def forward(self, x):
            return self.fc(x)

    TRANSLATION NOTES:
        - PyTorch Linear forward is x @ W^T + b.
        - Weight shape is (out_features, in_features), here (1,1).

    MOCK INJECTION:
        - x = jnp.array([[0.5], [1.0]], dtype=jnp.float32)
        - params = {"w": jnp.array([[2.0]], dtype=jnp.float32),
                    "b": jnp.array([1.0], dtype=jnp.float32)}
    """
    return jnp.dot(x, params["w"].T) + params["b"]


def generate_data(rng_key: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    PYTORCH EQUIVALENT:
        def generate_data():
            X = torch.rand(100, 1)
            y = 3 * X + 2 + torch.randn(100, 1) * 0.1
            return X, y

    TRANSLATION NOTES:
        - torch.rand -> jax.random.uniform
        - torch.randn -> jax.random.normal
        - RNG is explicit and split functionally.

    MOCK INJECTION:
        - X = jnp.linspace(0.0, 1.0, 100, dtype=jnp.float32).reshape(100, 1)
        - y = 3 * X + 2
    """
    k_x, k_noise = random.split(rng_key)
    X = random.uniform(k_x, (100, 1), minval=0.0, maxval=1.0, dtype=jnp.float32)
    noise = random.normal(k_noise, (100, 1), dtype=jnp.float32) * 0.1
    y = 3.0 * X + 2.0 + noise
    return X, y


def loss_fn(params: Dict[str, jax.Array], X: jax.Array, y: jax.Array) -> jax.Array:
    """
    PYTORCH EQUIVALENT:
        criterion = nn.MSELoss()
        predictions = model(X)
        loss = criterion(predictions, y)

    TRANSLATION NOTES:
        - nn.MSELoss() default reduction is mean.
        - This function takes params, X, y directly so gradients match
          the PyTorch train step structure closely.

    MOCK INJECTION:
        - params = {"w": jnp.array([[1.0]], dtype=jnp.float32),
                    "b": jnp.array([0.0], dtype=jnp.float32)}
        - X = jnp.array([[1.0], [2.0]], dtype=jnp.float32)
        - y = jnp.array([[2.0], [3.0]], dtype=jnp.float32)
    """
    predictions = model_apply(params, X)
    return jnp.mean((predictions - y) ** 2)


@jit
def train_step(
    params: Dict[str, jax.Array],
    X: jax.Array,
    y: jax.Array,
    lr: float,
):
    """
    PYTORCH EQUIVALENT:
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

    TRANSLATION NOTES:
        - optimizer.zero_grad() is omitted in JAX.
        - loss.backward() becomes jax.value_and_grad(loss_fn).
        - SGD update is manual: p = p - lr * grad.
        - No print() inside @jit.

    MOCK INJECTION:
        - params = {"w": jnp.array([[0.25]], dtype=jnp.float32),
                    "b": jnp.array([0.10], dtype=jnp.float32)}
        - X, y = fixed small tensors
    """
    loss_val, grads = jax.value_and_grad(loss_fn)(params, X, y)
    new_params = {
        "w": params["w"] - lr * grads["w"],
        "b": params["b"] - lr * grads["b"],
    }
    return new_params, loss_val


def train_model(
    X: jax.Array,
    y: jax.Array,
    params: Dict[str, jax.Array],
    num_epochs: int,
    lr: float,
) -> Dict[str, jax.Array]:
    """
    PYTORCH EQUIVALENT:
        def train_model(X, y, model, optimizer, criterion, num_epochs):
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                predictions = model(X)
                loss = criterion(predictions, y)
                loss.backward()
                optimizer.step()

    TRANSLATION NOTES:
        - Accepts X, y, params explicitly.
        - Does not create data or params internally.
        - Logging stays outside train_step.

    MOCK INJECTION:
        - Use fixed X, y, and params to test convergence.
    """
    for _ in range(num_epochs):
        params, _ = train_step(params, X, y, lr)
    return params


def save_params(params: Dict[str, jax.Array], path: str) -> None:
    params_np = {k: jax.device_get(v) for k, v in params.items()}
    with open(path, "wb") as f:
        pickle.dump(params_np, f)


def load_params(path: str) -> Dict[str, jax.Array]:
    with open(path, "rb") as f:
        params_np = pickle.load(f)
    return {k: jnp.array(v, dtype=jnp.float32) for k, v in params_np.items()}


def main() -> None:
    """
    PYTORCH EQUIVALENT:
        def main():
            torch.manual_seed(42)
            model = make_model()
            criterion = make_criterion()
            optimizer = make_optimizer(model)
            X, y = generate_data()
            epochs = 100
            train_model(X, y, model, optimizer, criterion, epochs)
            torch.save(model.state_dict(), "model.pth")
            loaded_model = make_model()
            loaded_model.load_state_dict(torch.load("model.pth"))
            loaded_model.eval()
            X_test = torch.tensor([[0.5], [1.0], [1.5]])
            with torch.no_grad():
                predictions = loaded_model(X_test)
                print(f"Predictions after loading: {predictions}")

    TRANSLATION NOTES:
        - torch.manual_seed(42) -> PRNGKey(42)
        - RNG order preserved conceptually:
            1) init params
            2) generate X
            3) generate noise for y

    MOCK INJECTION:
        - Replace params and data with fixed tensors for deterministic testing.
    """
    rng = random.PRNGKey(42)
    k_params, k_data = random.split(rng)

    params = init_params(k_params)
    X, y = generate_data(k_data)

    epochs = 100
    lr = 0.01
    params = train_model(X, y, params, epochs, lr)

    save_params(params, "model.pkl")
    loaded_params = load_params("model.pkl")

    X_test = jnp.array([[0.5], [1.0], [1.5]], dtype=jnp.float32)
    predictions = model_apply(loaded_params, X_test)
    print(f"Predictions after loading: {predictions}")


if __name__ == "__main__":
    main()