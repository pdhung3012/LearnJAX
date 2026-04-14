import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from matplotlib import pyplot as plt


def generate_data(key, n=100):
    """Generate synthetic linear data with noise.

    PYTORCH EQUIVALENT:
        def generate_data():
            X = torch.rand(100, 1) * 10
            y = 2 * X + 3 + torch.randn(100, 1)
            return X, y

    TRANSLATION NOTES:
        - torch.rand → jax.random.uniform
        - torch.randn → jax.random.normal
        - PRNGKey is split before each RNG use to avoid key reuse.

    MOCK INJECTION:
        Inject hardcoded X_np and y_np as jnp.array to bypass RNG.
    """
    key1, key2 = jax.random.split(key)
    X = jax.random.uniform(key1, shape=(n, 1)) * 10
    y = 2 * X + 3 + jax.random.normal(key2, shape=(n, 1))
    return X, y


def custom_activation(x):
    """Custom activation: tanh(x) + x.

    PYTORCH EQUIVALENT:
        def custom_activation(self, x):
            return torch.tanh(x) + x

    TRANSLATION NOTES:
        - torch.tanh → jnp.tanh. Pure function, no state needed.

    MOCK INJECTION:
        Pass any jnp array as x.
    """
    return jnp.tanh(x) + x


def model(params, X):
    """Forward pass: custom_activation(X @ w + b).

    PYTORCH EQUIVALENT:
        class CustomActivationModel(nn.Module):
            def __init__(self):
                self.linear = nn.Linear(1, 1)
            def custom_activation(self, x):
                return torch.tanh(x) + x
            def forward(self, x):
                return self.custom_activation(self.linear(x))

    TRANSLATION NOTES:
        - nn.Linear(1, 1) → jnp.dot(X, w) + b with params dict.
        - custom_activation is a standalone pure function.

    MOCK INJECTION:
        Inject params = {'w': W_np_as_jnp, 'b': B_np_as_jnp} with
        known W_np shape (1, 1) and B_np shape (1,).
    """
    return custom_activation(jnp.dot(X, params['w']) + params['b'])


def loss_fn(params, X, y):
    """MSE loss.

    PYTORCH EQUIVALENT:
        criterion = nn.MSELoss()
        loss = criterion(predictions, y)

    TRANSLATION NOTES:
        - nn.MSELoss() → jnp.mean((pred - y) ** 2). MSELoss default
          reduction='mean'.

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
        - nn.Linear uses Kaiming uniform for weight and uniform for bias.
          Here we use jax.random.uniform scaled to match PyTorch's
          kaiming_uniform_ default for fan_in=1: U(-1, 1).

    MOCK INJECTION:
        Bypass this entirely by constructing params dict from W_np, B_np.
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
        - loss.backward() + optimizer.step() → jax.value_and_grad + manual
          SGD param update.
        - optimizer.zero_grad() → not needed (JAX computes fresh grads).
        - optim.SGD(lr=0.01) → manual subtraction with lr=0.01.

    MOCK INJECTION:
        Inject params dict, X, y as hardcoded jnp arrays. Returns
        (new_params, loss) for comparison with PyTorch single-step output.
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
        - Logging (print) is outside @jit; train_step is jitted.
        - Returns final params for downstream use.

    MOCK INJECTION:
        Inject X, y, params as hardcoded jnp arrays. Compare returned
        params and logged losses against PyTorch run.
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
        - torch.manual_seed(42) → jax.random.PRNGKey(42)
        - Keys are split for each RNG-consuming call.
        - with torch.no_grad() → not needed in JAX.

    MOCK INJECTION:
        Not applicable — main() is the top-level driver.
    """
    key = jax.random.PRNGKey(42)
    key, subkey1, subkey2 = jax.random.split(key, 3)

    X, y = generate_data(subkey1)
    params = init_params(subkey2)

    params = train_model(X, y, params, num_epochs=1000, lr=0.01)

    print(f"Learned weight: {params['w'].item():.4f}, Learned bias: {params['b'].item():.4f}")

    plt.figure(figsize=(4, 4))
    plt.scatter(X, y, label='Training Data')
    plt.plot(X, params['w'].item() * X + params['b'].item(), 'r', label='Model Fit')
    plt.legend()
    plt.show()

    X_test = jnp.array([[4.0], [7.0]])
    predictions = model(params, X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")


if __name__ == '__main__':
    main()
