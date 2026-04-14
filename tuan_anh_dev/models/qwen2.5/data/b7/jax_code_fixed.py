import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from torch.utils.tensorboard import SummaryWriter


def generate_data(key, n=100):
    """Generate synthetic linear data with noise.

    PYTORCH EQUIVALENT:
        def generate_data():
            X = torch.rand(100, 1) * 10
            y = 3 * X + 5 + torch.randn(100, 1)
            return X, y

    TRANSLATION NOTES:
        - torch.rand → jax.random.uniform, torch.randn → jax.random.normal.
        - Note: coefficients are y = 3*X + 5 (different from b1-b5).
        - Keys split before each RNG use.

    MOCK INJECTION:
        Inject hardcoded X_np and y_np as jnp.array to bypass RNG.
    """
    key1, key2 = jax.random.split(key)
    X = jax.random.uniform(key1, shape=(n, 1)) * 10
    y = 3 * X + 5 + jax.random.normal(key2, shape=(n, 1))
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
    """One SGD training step.

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


def train_model(X, y, params, num_epochs, lr=0.01, writer=None):
    """Full training loop with TensorBoard logging.

    PYTORCH EQUIVALENT:
        def train_model(X, y, model, optimizer, criterion, num_epochs, writer):
            for epoch in range(num_epochs):
                predictions = model(X)
                loss = criterion(predictions, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                writer.add_scalar("Loss/train", loss.item(), epoch)
                if (epoch + 1) % 10 == 0:
                    print(...)

    TRANSLATION NOTES:
        - Accepts (X, y, params, ...) — never generates data or params
          internally.
        - writer.add_scalar is kept outside @jit (Python side effect).
        - TensorBoard SummaryWriter is reused from torch.utils.tensorboard
          since it works with plain Python scalars — no PyTorch dependency
          needed at runtime beyond tensorboard.
        - print() every 10 epochs (not 100) to match PyTorch original.

    MOCK INJECTION:
        Inject X, y, params. Pass writer=None to skip TensorBoard logging
        in tests.
    """
    for epoch in range(num_epochs):
        params, loss = train_step(params, X, y, lr)
        if writer is not None:
            writer.add_scalar("Loss/train", loss.item(), epoch)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    return params


def main():
    """Main entry point mirroring PyTorch main().

    PYTORCH EQUIVALENT:
        def main():
            torch.manual_seed(42)
            X, y = generate_data()
            writer = SummaryWriter(log_dir="runs/linear_regression")
            model = make_model()
            criterion = make_criterion()
            optimizer = make_optimizer(model)
            train_model(X, y, model, optimizer, criterion, num_epochs=100,
                        writer=writer)
            writer.close()

    TRANSLATION NOTES:
        - torch.manual_seed(42) → jax.random.PRNGKey(42).
        - Writer created before model init to match PyTorch execution order.
        - Keys split for each RNG-consuming call.

    MOCK INJECTION:
        Not applicable — main() is the top-level driver.
    """
    key = jax.random.PRNGKey(42)
    key, subkey1, subkey2 = jax.random.split(key, 3)

    X, y = generate_data(subkey1)
    # Writer created before model init to match PyTorch execution order
    writer = SummaryWriter(log_dir="runs/linear_regression")
    params = init_params(subkey2)

    params = train_model(X, y, params, num_epochs=100, lr=0.01, writer=writer)
    writer.close()


if __name__ == '__main__':
    main()
