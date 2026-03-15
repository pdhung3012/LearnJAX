import jax
import jax.numpy as jnp
from jax import random
from typing import Callable, Dict, Tuple


def generate_data(rng_key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    PYTORCH EQUIVALENT:
        X = torch.rand(100, 1) * 10
        y = 2 * X + 3 + torch.randn(100, 1)
        return X, y

    TRANSLATION NOTES:
        - torch.manual_seed(42) sets global RNG state; in JAX we pass an explicit rng_key in.
        - torch.rand and torch.randn consume RNG state in-order; we preserve that by splitting
          rng_key into two keys (one for uniform, one for normal) and using them in that order.
        - Shapes and scaling match PyTorch: (100, 1), uniform in [0,1) then *10, normal mean 0 std 1.

    MOCK INJECTION:
        - For deterministic unit tests without RNG, inject:
          X = jnp.array([[0.0], [1.0]], dtype=jnp.float32)
          y = 2 * X + 3 + jnp.array([[0.0], [0.0]], dtype=jnp.float32)
        - Or pass a fixed rng_key = random.PRNGKey(42) and assert exact shapes/statistics.
    """
    key_u, key_n = random.split(rng_key, 2)
    X = random.uniform(key_u, shape=(100, 1)) * 10.0
    y = 2.0 * X + 3.0 + random.normal(key_n, shape=(100, 1))
    return X, y


def model_apply(params: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    """
    PYTORCH EQUIVALENT:
        class LinearRegressionModel(nn.Module):
            def __init__(...):
                self.linear = nn.Linear(1, 1)
            def forward(self, x):
                return self.linear(x)

    TRANSLATION NOTES:
        - nn.Module is translated to a pure function with an explicit params dict.
        - nn.Linear(1,1) maps to y = x @ w + b with:
            w shape (in_features, out_features) = (1, 1)
            b shape (out_features,) = (1,)
        - We keep the same broadcasting behavior as PyTorch's Linear: (N,1) @ (1,1) -> (N,1),
          then + (1,) broadcasts to (N,1).

    MOCK INJECTION:
        - Inject params = {"w": jnp.array([[2.0]]), "b": jnp.array([3.0])}
          and x = jnp.array([[4.0], [7.0]]) to expect [[11.0],[17.0]] (no noise).
    """
    return jnp.dot(x, params["w"]) + params["b"]


def make_model(rng_key: jax.Array) -> Dict[str, jnp.ndarray]:
    """
    PYTORCH EQUIVALENT:
        model = LinearRegressionModel()
        # nn.Linear uses PyTorch default initialization internally.

    TRANSLATION NOTES:
        - PyTorch's nn.Linear default init is framework-specific and not identical to JAX defaults.
          Here we implement a reasonable JAX init to enable training, but note it will NOT match
          PyTorch's exact initial weights unless you replicate PyTorch init exactly.
        - We keep parameter shapes identical to nn.Linear(1,1): w (1,1), b (1,).
        - RNG is explicit and kept inside this function because PyTorch initializes weights
          at model construction time.

    MOCK INJECTION:
        - For tests, bypass RNG and inject:
          params = {"w": jnp.array([[0.0]]), "b": jnp.array([0.0])}
        - Or use rng_key = random.PRNGKey(0) and snapshot params shapes/dtypes.
    """
    # NOTE: This does NOT replicate PyTorch nn.Linear exact init distribution/values.
    # It preserves the structural role of "random init at model construction".
    key_w, key_b = random.split(rng_key, 2)
    w = random.normal(key_w, shape=(1, 1)) * 0.01
    b = random.normal(key_b, shape=(1,)) * 0.01
    return {"w": w, "b": b}


def make_criterion() -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    PYTORCH EQUIVALENT:
        criterion = nn.MSELoss()

    TRANSLATION NOTES:
        - nn.MSELoss() defaults to mean reduction in PyTorch.
        - We implement mean squared error: mean((pred - y)**2) matching that default.

    MOCK INJECTION:
        - Inject pred = jnp.array([[0.0],[2.0]]), y = jnp.array([[1.0],[1.0]])
          to get mean([(1)^2,(1)^2]) = 1.0.
    """
    def mse_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean((pred - target) ** 2)
    return mse_loss


def make_optimizer(lr: float = 0.01) -> Dict[str, float]:
    """
    PYTORCH EQUIVALENT:
        optimizer = optim.SGD(model.parameters(), lr=0.01)

    TRANSLATION NOTES:
        - PyTorch optim.SGD with only lr specified corresponds to plain SGD (no momentum, no weight decay).
        - In JAX we store optimizer hyperparameters as a small dict; updates are done manually.

    MOCK INJECTION:
        - Inject optimizer = {"lr": 0.01} and verify a single SGD step:
          new = old - lr * grad for each param tensor.
    """
    return {"lr": lr}


def loss_fn(
    params: Dict[str, jnp.ndarray],
    X: jnp.ndarray,
    y: jnp.ndarray,
    criterion: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """
    PYTORCH EQUIVALENT:
        predictions = model(X)
        loss = criterion(predictions, y)

    TRANSLATION NOTES:
        - Forward pass is model_apply(params, X).
        - Criterion is passed in (like PyTorch object) and applied the same way.
        - Separate loss_fn makes value_and_grad straightforward.

    MOCK INJECTION:
        - params={"w":[[2.0]],"b":[3.0]}, X=[[1.0]], y=[[5.0]] => pred=5.0 => loss=0.0
    """
    preds = model_apply(params, X)
    return criterion(preds, y)


# FIX ONLY: use functools.partial so jax.jit receives the function argument.
import functools

@functools.partial(jax.jit, static_argnames=("criterion",))
def train_step(
    params: Dict[str, jnp.ndarray],
    X: jnp.ndarray,
    y: jnp.ndarray,
    optimizer: Dict[str, float],
    criterion: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
    """
    PYTORCH EQUIVALENT:
        predictions = model(X)
        loss = criterion(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    TRANSLATION NOTES:
        - optimizer.zero_grad() is omitted: JAX gradients are computed fresh each step.
        - loss.backward() + optimizer.step() becomes:
            grads = grad(loss_fn)(params, X, y, criterion)
            params = params - lr * grads  (per-tensor)
        - @jit is applied here only; no prints or side effects inside this function.

    MOCK INJECTION:
        - Use tiny inputs:
          X=jnp.array([[1.0]],dtype=jnp.float32), y=jnp.array([[5.0]])
          params={"w":jnp.array([[0.0]]),"b":jnp.array([0.0])}
          optimizer={"lr":0.01}
          Verify loss decreases after one step (loss before > loss after).
    """
    # Return the *pre-update* loss to match PyTorch's `pt_loss_tensor` in the test.
    loss_before = loss_fn(params, X, y, criterion)

    # Compute grads at the same (pre-update) params.
    _, grads = jax.value_and_grad(loss_fn)(params, X, y, criterion)

    lr = optimizer["lr"]
    new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    return new_params, loss_before


def train_model(
    X: jnp.ndarray,
    y: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    optimizer: Dict[str, float],
    criterion: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    num_epochs: int,
) -> Dict[str, jnp.ndarray]:
    """
    PYTORCH EQUIVALENT:
        for epoch in range(epochs):
            predictions = model(X)
            loss = criterion(predictions, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    TRANSLATION NOTES:
        - This function accepts X, y, params as arguments and does NOT create them internally.
        - Logging stays here (not in @jit train_step), matching the PyTorch print cadence and format.
        - loss_val is a scalar DeviceArray; we convert to float for formatting.

    MOCK INJECTION:
        - Inject X=jnp.array([[0.0],[1.0]]), y=2*X+3 (no noise),
          params={"w":[[0.0]],"b":[0.0]}, epochs=200
          Expect prints at epochs 100 and 200 and parameters move toward w≈2, b≈3.
    """
    for epoch in range(num_epochs):
        params, loss_val = train_step(params, X, y, optimizer, criterion)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {float(loss_val):.4f}")

    return params


def main():
    """
    PYTORCH EQUIVALENT:
        torch.manual_seed(42)
        X, y = generate_data()
        model = make_model()
        criterion = make_criterion()
        optimizer = make_optimizer(model)
        epochs = 1000
        train_model(X, y, model, optimizer, criterion, epochs)
        [w, b] = model.linear.parameters()
        print(...)
        X_test = torch.tensor([[4.0],[7.0]])
        with torch.no_grad():
            predictions = model(X_test)
            print(...)

    TRANSLATION NOTES:
        - torch.manual_seed(42) maps to rng = random.PRNGKey(42).
        - To preserve PyTorch's RNG call ordering (seed -> data gen -> model init),
          we split rng in that same sequence:
            key_data used by generate_data
            key_model used by make_model
        - Criterion and optimizer are created after model, matching original order.
        - No need for torch.no_grad in JAX; we just call model_apply for inference.
        - Printed outputs (loss values and learned parameters) will generally differ
          numerically from PyTorch unless you exactly replicate PyTorch init and RNG.
          Structure and logic are mirrored 1-to-1.

    MOCK INJECTION:
        - Replace RNG-dependent parts for golden tests:
            X_test = jnp.array([[4.0],[7.0]], dtype=jnp.float32)
            params={"w":jnp.array([[2.0]]),"b":jnp.array([3.0])}
          and assert predictions == [[11.0],[17.0]].
    """
    rng = random.PRNGKey(42)

    # Preserve call order: seed -> generate_data -> make_model -> make_criterion -> make_optimizer
    rng, key_data = random.split(rng, 2)
    X, y = generate_data(key_data)

    rng, key_model = random.split(rng, 2)
    params = make_model(key_model)

    criterion = make_criterion()
    optimizer = make_optimizer(lr=0.01)

    epochs = 1000
    params = train_model(X, y, params, optimizer, criterion, epochs)

    # Display learned parameters (match PyTorch's "w.item()" and "b.item()" formatting)
    w = params["w"]
    b = params["b"]
    print(f"Learned weight: {float(w.reshape(-1)[0]):.4f}, Learned bias: {float(b.reshape(-1)[0]):.4f}")

    # Testing on new data
    X_test = jnp.array([[4.0], [7.0]], dtype=jnp.float32)
    predictions = model_apply(params, X_test)

    # Match PyTorch-ish printing style using Python lists
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")


if __name__ == "__main__":
    main()