from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import optax


def generate_data(key, n=100):
    """Generate synthetic data: y = x0 + 2*x1 + noise.

    PYTORCH EQUIVALENT:
        def generate_data():
            X = torch.rand(100, 2) * 10
            y = (X[:, 0] + X[:, 1] * 2).unsqueeze(1) + torch.randn(100, 1)
            return X, y

    TRANSLATION NOTES:
        - torch.rand → jax.random.uniform, torch.randn → jax.random.normal.
        - unsqueeze(1) → jnp expansion via [:, None].
        - Keys are split before each RNG use.

    MOCK INJECTION:
        Inject hardcoded X_np (100,2) and y_np (100,1) as jnp.array.
    """
    key1, key2 = jax.random.split(key)
    X = jax.random.uniform(key1, shape=(n, 2)) * 10
    y = (X[:, 0] + X[:, 1] * 2)[:, None] + jax.random.normal(key2, shape=(n, 1))
    return X, y


def model(params, X):
    """Forward pass: two-layer DNN with ReLU.

    PYTORCH EQUIVALENT:
        class DNNModel(nn.Module):
            def __init__(self):
                self.fc1 = nn.Linear(2, 10)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(10, 1)
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x

    TRANSLATION NOTES:
        - nn.Linear → jnp.dot(x, w) + b for each layer.
        - nn.ReLU() → jax.nn.relu.
        - params dict has keys 'w1', 'b1', 'w2', 'b2'.

    MOCK INJECTION:
        Inject params = {'w1': (2,10), 'b1': (10,), 'w2': (10,1), 'b2': (1,)}.
    """
    x = jnp.dot(X, params['w1']) + params['b1']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['w2']) + params['b2']
    return x


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
    """Initialise params for a 2-layer DNN: Linear(2,10) → ReLU → Linear(10,1).

    PYTORCH EQUIVALENT:
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)

    TRANSLATION NOTES:
        - nn.Linear default init is Kaiming uniform for weights and
          uniform for bias. Approximated here with jax.random.uniform
          scaled by 1/sqrt(fan_in).

    MOCK INJECTION:
        Bypass by constructing params dict from W1_np, B1_np, W2_np, B2_np.
    """
    key1, key2, key3, key4 = jax.random.split(key, 4)
    w1 = jax.random.uniform(key1, shape=(2, 10), minval=-1.0 / jnp.sqrt(2.0),
                            maxval=1.0 / jnp.sqrt(2.0))
    b1 = jax.random.uniform(key2, shape=(10,), minval=-1.0 / jnp.sqrt(2.0),
                            maxval=1.0 / jnp.sqrt(2.0))
    w2 = jax.random.uniform(key3, shape=(10, 1), minval=-1.0 / jnp.sqrt(10.0),
                            maxval=1.0 / jnp.sqrt(10.0))
    b2 = jax.random.uniform(key4, shape=(1,), minval=-1.0 / jnp.sqrt(10.0),
                            maxval=1.0 / jnp.sqrt(10.0))
    return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}


@partial(jit, static_argnums=(4,))
def train_step(params, X, y, opt_state, optimizer):
    """One Adam training step.

    PYTORCH EQUIVALENT:
        predictions = model(X)
        loss = criterion(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    TRANSLATION NOTES:
        - loss.backward() + optimizer.step() → value_and_grad + optax update.
        - optim.Adam(lr=0.01) → optax.adam(0.01).
        - optimizer.zero_grad() → not needed.

    MOCK INJECTION:
        Inject params, X, y, opt_state. Returns (new_params, opt_state, loss).
    """
    loss, grads = value_and_grad(loss_fn)(params, X, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss


def train_model(X, y, params, num_epochs, optimizer, opt_state):
    """Full training loop with Adam optimizer.

    PYTORCH EQUIVALENT:
        def train_model(X, y, model, optimizer, criterion, num_epochs):
            for epoch in range(num_epochs):
                predictions = model(X)
                loss = criterion(predictions, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (epoch + 1) % 100 == 0:
                    print(...)

    TRANSLATION NOTES:
        - Accepts (X, y, params, ...) — never generates data or params
          internally.
        - print() is outside @jit.
        - Returns final params and opt_state.

    MOCK INJECTION:
        Inject X, y, params, opt_state. Compare returned params against
        PyTorch run.
    """
    for epoch in range(num_epochs):
        params, opt_state, loss = train_step(params, X, y, opt_state, optimizer)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    return params, opt_state


def main():
    """Main entry point mirroring PyTorch main().

    PYTORCH EQUIVALENT:
        def main():
            torch.manual_seed(42)
            X, y = generate_data()
            model = make_model()
            criterion = make_criterion()
            optimizer = make_optimizer(model)
            train_model(X, y, model, optimizer, criterion, num_epochs=1000)
            ...

    TRANSLATION NOTES:
        - torch.manual_seed(42) → jax.random.PRNGKey(42).
        - optim.Adam(lr=0.01) → optax.adam(0.01).
        - with torch.no_grad() → not needed.

    MOCK INJECTION:
        Not applicable — main() is the top-level driver.
    """
    key = jax.random.PRNGKey(42)
    key, subkey1, subkey2 = jax.random.split(key, 3)

    X, y = generate_data(subkey1)
    params = init_params(subkey2)

    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(params)

    params, opt_state = train_model(X, y, params, num_epochs=1000,
                                    optimizer=optimizer, opt_state=opt_state)

    X_test = jnp.array([[4.0, 3.0], [7.0, 8.0]])
    predictions = model(params, X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")


if __name__ == '__main__':
    main()
