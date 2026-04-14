import jax
import jax.numpy as jnp
from jax import jit, value_and_grad


def generate_data(key, n=100):
    """Generate synthetic linear data with noise.

    PYTORCH EQUIVALENT:
        def generate_data():
            X = torch.rand(100, 1) * 10
            y = 2 * X + 3 + torch.randn(100, 1)
            return X, y

    TRANSLATION NOTES:
        - torch.rand → jax.random.uniform, torch.randn → jax.random.normal.
        - Keys split before each RNG use.

    MOCK INJECTION:
        Inject hardcoded X_np and y_np as jnp.array to bypass RNG.
    """
    key1, key2 = jax.random.split(key)
    X = jax.random.uniform(key1, shape=(n, 1)) * 10
    y = 2 * X + 3 + jax.random.normal(key2, shape=(n, 1))
    return X, y


def model(params, X):
    """Forward pass: X @ w + b (simple linear regression).

    PYTORCH EQUIVALENT:
        class LinearRegressionModel(nn.Module):
            def __init__(self):
                self.linear = nn.Linear(1, 1)
            def forward(self, x):
                return self.linear(x)

    TRANSLATION NOTES:
        - nn.Linear(1, 1) → jnp.dot(X, w) + b with params dict.

    MOCK INJECTION:
        Inject params = {'w': W_np_as_jnp, 'b': B_np_as_jnp}.
    """
    return jnp.dot(X, params['w']) + params['b']


def huber_loss(y_pred, y_true, delta=1.0):
    """Huber loss with configurable delta.

    PYTORCH EQUIVALENT:
        class HuberLoss(nn.Module):
            def __init__(self, delta=1.0):
                super().__init__()
                self.delta = delta
            def forward(self, y_pred, y_true):
                error = torch.abs(y_pred - y_true)
                loss = torch.where(error <= self.delta,
                                   0.5 * error**2,
                                   self.delta * (error - 0.5 * self.delta))
                return loss.mean()

    TRANSLATION NOTES:
        - torch.abs → jnp.abs, torch.where → jnp.where.
        - loss.mean() → jnp.mean(loss).
        - Class with self.delta → plain function with delta argument.

    MOCK INJECTION:
        Pass hardcoded y_pred and y_true jnp arrays.
    """
    error = jnp.abs(y_pred - y_true)
    loss = jnp.where(error <= delta,
                     0.5 * error ** 2,
                     delta * (error - 0.5 * delta))
    return jnp.mean(loss)


def loss_fn(params, X, y, delta=1.0):
    """Compute Huber loss for model predictions.

    PYTORCH EQUIVALENT:
        criterion = HuberLoss(delta=1.0)
        loss = criterion(predictions, y)

    TRANSLATION NOTES:
        - Combines forward pass and loss into one differentiable function.
        - delta=1.0 matches the PyTorch HuberLoss default.

    MOCK INJECTION:
        Inject params, X, y as hardcoded jnp arrays.
    """
    predictions = model(params, X)
    return huber_loss(predictions, y, delta)


def init_params(key):
    """Initialise weight and bias matching nn.Linear(1, 1) default init.

    PYTORCH EQUIVALENT:
        self.linear = nn.Linear(1, 1)

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
    """One SGD training step with Huber loss.

    PYTORCH EQUIVALENT:
        predictions = model(X)
        loss = criterion(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    TRANSLATION NOTES:
        - loss.backward() + optimizer.step() → value_and_grad + manual SGD.
        - optimizer.zero_grad() → not needed.
        - optim.SGD(lr=0.01) → manual subtraction with lr=0.01.

    MOCK INJECTION:
        Inject params, X, y. Returns (new_params, loss).
    """
    loss, grads = value_and_grad(loss_fn)(params, X, y)
    new_params = {k: params[k] - lr * grads[k] for k in params}
    return new_params, loss


def train_model(X, y, params, num_epochs, lr=0.01):
    """Full training loop.

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

    MOCK INJECTION:
        Inject X, y, params. Compare returned params and losses against
        PyTorch run.
    """
    for epoch in range(num_epochs):
        params, loss = train_step(params, X, y, lr)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    return params


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
        - Keys split for each RNG-consuming call.
        - with torch.no_grad() → not needed.

    MOCK INJECTION:
        Not applicable — main() is the top-level driver.
    """
    key = jax.random.PRNGKey(42)
    key, subkey1, subkey2 = jax.random.split(key, 3)

    X, y = generate_data(subkey1)
    params = init_params(subkey2)

    params = train_model(X, y, params, num_epochs=1000, lr=0.01)

    print(f"Learned weight: {params['w'].item():.4f}, Learned bias: {params['b'].item():.4f}")

    X_test = jnp.array([[4.0], [7.0]])
    predictions = model(params, X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")


if __name__ == '__main__':
    main()
