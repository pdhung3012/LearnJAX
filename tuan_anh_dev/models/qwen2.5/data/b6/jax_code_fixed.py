import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import pickle


def generate_data(key, n=100):
    """Generate synthetic linear data with noise.

    PYTORCH EQUIVALENT:
        def generate_data():
            X = torch.rand(100, 1)
            y = 3 * X + 2 + torch.randn(100, 1) * 0.1
            return X, y

    TRANSLATION NOTES:
        - torch.rand → jax.random.uniform, torch.randn → jax.random.normal.
        - Note: X range is [0, 1) (no *10), noise scale is 0.1.
        - Keys split before each RNG use.

    MOCK INJECTION:
        Inject hardcoded X_np and y_np as jnp.array to bypass RNG.
    """
    key1, key2 = jax.random.split(key)
    X = jax.random.uniform(key1, shape=(n, 1))
    y = 3 * X + 2 + jax.random.normal(key2, shape=(n, 1)) * 0.1
    return X, y


def model(params, X):
    """Forward pass: X @ w + b (simple linear regression).

    PYTORCH EQUIVALENT:
        class SimpleModel(nn.Module):
            def __init__(self):
                self.fc = nn.Linear(1, 1)
            def forward(self, x):
                return self.fc(x)

    TRANSLATION NOTES:
        - nn.Linear(1, 1) → jnp.dot(X, w) + b with params dict.

    MOCK INJECTION:
        Inject params = {'w': W_np_as_jnp, 'b': B_np_as_jnp}.
    """
    return jnp.dot(X, params['w']) + params['b']


def loss_fn(params, X, y):
    """MSE loss.

    PYTORCH EQUIVALENT:
        criterion = nn.MSELoss()
        loss = criterion(predictions, y)

    TRANSLATION NOTES:
        - nn.MSELoss() → jnp.mean((pred - y) ** 2).

    MOCK INJECTION:
        Inject params, X, y as hardcoded jnp arrays.
    """
    predictions = model(params, X)
    return jnp.mean((predictions - y) ** 2)


def init_params(key):
    """Initialise weight and bias matching nn.Linear(1, 1) default init.

    PYTORCH EQUIVALENT:
        self.fc = nn.Linear(1, 1)

    TRANSLATION NOTES:
        - nn.Linear Kaiming uniform for fan_in=1 → U(-1, 1).

    MOCK INJECTION:
        Bypass by constructing params dict from W_np, B_np.
    """
    key1, key2 = jax.random.split(key)
    w = jax.random.uniform(key1, shape=(1, 1), minval=-1.0, maxval=1.0)
    b = jax.random.uniform(key2, shape=(1,), minval=-1.0, maxval=1.0)
    return {'w': w, 'b': b}


@jit
def train_step(params, X, y, lr=0.01):
    """One SGD training step.

    PYTORCH EQUIVALENT:
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

    TRANSLATION NOTES:
        - Note: PyTorch b6 calls zero_grad() before forward pass (different
          order from other examples). Result is identical since grads are
          freshly computed either way.
        - loss.backward() + optimizer.step() → value_and_grad + manual SGD.

    MOCK INJECTION:
        Inject params, X, y. Returns (new_params, loss).
    """
    loss, grads = value_and_grad(loss_fn)(params, X, y)
    new_params = {k: params[k] - lr * grads[k] for k in params}
    return new_params, loss


def train_model(X, y, params, num_epochs, lr=0.01):
    """Full training loop (no logging — matches PyTorch original).

    PYTORCH EQUIVALENT:
        def train_model(X, y, model, optimizer, criterion, num_epochs):
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                predictions = model(X)
                loss = criterion(predictions, y)
                loss.backward()
                optimizer.step()

    TRANSLATION NOTES:
        - Accepts (X, y, params, ...) — never generates data or params
          internally.
        - No print() in PyTorch original, so none here either.
        - Returns final params.

    MOCK INJECTION:
        Inject X, y, params. Compare returned params against PyTorch run.
    """
    for epoch in range(num_epochs):
        params, loss = train_step(params, X, y, lr)
    return params


def main():
    """Main entry point mirroring PyTorch main().

    PYTORCH EQUIVALENT:
        def main():
            torch.manual_seed(42)
            model = make_model()
            ...
            X, y = generate_data()
            train_model(...)
            torch.save(model.state_dict(), "model.pth")
            loaded_model = SimpleModel()
            loaded_model.load_state_dict(torch.load("model.pth"))
            ...

    TRANSLATION NOTES:
        - torch.manual_seed(42) → jax.random.PRNGKey(42).
        - Model init before data generation (matches PyTorch RNG order).
        - torch.save/load_state_dict → pickle.dump/pickle.load.
        - loaded_model.eval() → not needed (no batch norm / dropout).
        - with torch.no_grad() → not needed.

    MOCK INJECTION:
        Not applicable — main() is the top-level driver.
    """
    key = jax.random.PRNGKey(42)
    key, subkey1, subkey2 = jax.random.split(key, 3)

    # Model init before data generation to match PyTorch RNG order
    params = init_params(subkey1)
    X, y = generate_data(subkey2)

    params = train_model(X, y, params, num_epochs=100, lr=0.01)

    # Save params
    with open("model.pkl", "wb") as f:
        pickle.dump(params, f)

    # Load params
    with open("model.pkl", "rb") as f:
        loaded_params = pickle.load(f)

    X_test = jnp.array([[0.5], [1.0], [1.5]])
    predictions = model(loaded_params, X_test)
    print(f"Predictions after loading: {predictions}")


if __name__ == '__main__':
    main()
