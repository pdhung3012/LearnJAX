import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad


def generate_data(key):
    """Generate synthetic data. Returns (X, y).

    PYTORCH EQUIVALENT:
        torch.manual_seed(42)
        X = torch.rand(100, 1) * 10
        y = 2 * X + 3 + torch.randn(100, 1)

    TRANSLATION NOTES:
        - PyTorch's torch.rand maps to jax.random.uniform (both produce U[0,1)).
        - PyTorch's torch.randn maps to jax.random.normal (both produce N(0,1)).
        - The global torch.manual_seed(42) is replaced by an explicit PRNGKey
          passed in from the caller. We split the key once to get independent
          streams for X (uniform) and noise (normal), mirroring the two
          distinct RNG draws in the PyTorch version.
        - Numeric results will differ from PyTorch because the underlying
          PRNG algorithms are different; structural equivalence is preserved.

    MOCK INJECTION:
        Pass deterministic arrays to bypass RNG entirely:
            X = jnp.array([[1.0], [2.0], [3.0]])
            y = jnp.array([[5.0], [7.0], [9.0]])
    """
    key_x, key_noise = random.split(key)
    X = random.uniform(key_x, shape=(100, 1)) * 10  # 100 data points between 0 and 10
    y = 2 * X + 3 + random.normal(key_noise, shape=(100, 1))  # Linear relationship with noise
    return X, y


def init_params(key):
    """Initialise model parameters as a plain dict.

    PYTORCH EQUIVALENT:
        class LinearRegressionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)

        model = LinearRegressionModel()

    TRANSLATION NOTES:
        - nn.Linear(1, 1) is replaced by an explicit params dict with 'w'
          of shape (1, 1) and 'b' of shape (1,).
        - PyTorch's nn.Linear default init uses kaiming_uniform for weight
          and uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) for bias. For a
          single-feature layer (fan_in=1) this is U(-1, 1) for both.
          We replicate this with jax.random.uniform over [-1, 1).
        - A dedicated key is split for w and b to keep the two draws
          independent, matching the two separate init draws in PyTorch.

    MOCK INJECTION:
        Supply a fixed params dict to skip random init:
            params = {'w': jnp.array([[0.5]]), 'b': jnp.array([0.0])}
    """
    key_w, key_b = random.split(key)
    w = random.uniform(key_w, shape=(1, 1), minval=-1.0, maxval=1.0)
    b = random.uniform(key_b, shape=(1,), minval=-1.0, maxval=1.0)
    return {'w': w, 'b': b}


def linear_model(params, x):
    """Forward pass: single linear layer.

    PYTORCH EQUIVALENT:
        def forward(self, x):
            return self.linear(x)

    TRANSLATION NOTES:
        - nn.Linear(in, out) computes x @ weight.T + bias. For shapes
          x:(N,1), w:(1,1), b:(1,), jnp.dot(x, w) + b is equivalent.
        - No activation — identical to the PyTorch model.

    MOCK INJECTION:
        params = {'w': jnp.array([[2.0]]), 'b': jnp.array([3.0])}
        x = jnp.array([[1.0], [2.0]])
        # Expected output: [[5.0], [7.0]]
    """
    return jnp.dot(x, params['w']) + params['b']


def mse_loss(params, X, y):
    """Mean squared error loss.

    PYTORCH EQUIVALENT:
        criterion = nn.MSELoss()
        loss = criterion(predictions, y)

    TRANSLATION NOTES:
        - nn.MSELoss() defaults to reduction='mean', so we use jnp.mean
          over the squared differences.
        - The forward pass is called inside this function so that
          jax.value_and_grad can differentiate through both the model
          and the loss in a single call.

    MOCK INJECTION:
        params = {'w': jnp.array([[2.0]]), 'b': jnp.array([3.0])}
        X = jnp.array([[1.0]]); y = jnp.array([[5.0]])
        # prediction = 5.0, loss = 0.0
    """
    predictions = linear_model(params, X)
    return jnp.mean((predictions - y) ** 2)


@jit
def train_step(params, X, y, lr):
    """Single training step: compute loss and update params via SGD.

    PYTORCH EQUIVALENT:
        predictions = model(X)
        loss = criterion(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    TRANSLATION NOTES:
        - jax.value_and_grad replaces the explicit backward() call and
          returns both the scalar loss and the gradient dict.
        - optimizer.zero_grad() is omitted; JAX computes fresh gradients
          each call (functional style, no accumulated state).
        - optim.SGD(lr=0.01) is translated to the manual update rule
          param = param - lr * grad. The learning rate is passed in so
          this function stays pure.
        - @jit is applied here as required. No print() or other Python
          side effects are present inside this function.

    MOCK INJECTION:
        params = {'w': jnp.array([[0.0]]), 'b': jnp.array([0.0])}
        X = jnp.array([[1.0]]); y = jnp.array([[5.0]])
        lr = 0.01
        # Verify that returned loss is 25.0 and params shift toward target.
    """
    loss, grads = value_and_grad(mse_loss)(params, X, y)
    updated_params = {k: params[k] - lr * grads[k] for k in params}
    return updated_params, loss


def train_model(X, y, params, lr, num_epochs):
    """Full training loop with logging.

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
        - The signature accepts (X, y, params, ...) instead of
          (X, y, model, optimizer, criterion, ...) because JAX uses a
          functional style: params replaces the stateful model, lr replaces
          the optimizer object, and the loss function is called inside
          train_step rather than passed as a separate argument.
        - Logging (print) is kept here, outside the @jit boundary, to
          avoid side-effect issues with JAX tracing.
        - train_step is the @jit-compiled inner function that performs
          the actual forward/backward/update.

    MOCK INJECTION:
        X = jnp.array([[1.0], [2.0]])
        y = jnp.array([[5.0], [7.0]])
        params = {'w': jnp.array([[0.0]]), 'b': jnp.array([0.0])}
        lr = 0.01; num_epochs = 100
        # Verify loss decreases monotonically and params converge.
    """
    for epoch in range(num_epochs):
        params, loss = train_step(params, X, y, lr)

        # Log progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    return params


def main():
    """Entry point mirroring the PyTorch main().

    PYTORCH EQUIVALENT:
        def main():
            torch.manual_seed(42)
            X, y = generate_data()
            model = make_model()
            criterion = make_criterion()
            optimizer = make_optimizer(model)
            train_model(X, y, model, optimizer, criterion, num_epochs=1000)
            [w, b] = model.linear.parameters()
            print(...)
            X_test = torch.tensor([[4.0], [7.0]])
            with torch.no_grad():
                predictions = model(X_test)
                print(...)

    TRANSLATION NOTES:
        - torch.manual_seed(42) becomes jax.random.PRNGKey(42). The key
          is split to feed generate_data and init_params with independent
          sub-keys, preserving the original seed → data → model ordering.
        - make_criterion() and make_optimizer() have no direct JAX
          counterparts; the loss is hard-coded in mse_loss and the
          learning rate (0.01) is passed directly to train_model.
        - 'with torch.no_grad()' is omitted; JAX only computes gradients
          when explicitly requested via jax.grad or jax.value_and_grad.
        - Numeric outputs will differ from PyTorch due to different PRNG
          algorithms, but the structure and convergence behaviour are
          equivalent.

    MOCK INJECTION:
        Replace generate_data and init_params with fixed tensors:
            X = jnp.ones((100, 1)) * 5.0
            y = jnp.ones((100, 1)) * 13.0
            params = {'w': jnp.array([[0.0]]), 'b': jnp.array([0.0])}
    """
    # RNG key at top of main, matching torch.manual_seed(42) placement.
    key = random.PRNGKey(42)
    key_data, key_model = random.split(key)

    # Call order mirrors original: data -> params -> train -> display -> test.
    X, y = generate_data(key_data)
    params = init_params(key_model)

    lr = 0.01  # Same as optim.SGD(lr=0.01)
    params = train_model(X, y, params, lr, num_epochs=1000)

    # Display the learned parameters
    w = params['w']
    b = params['b']
    print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

    # Testing on new data (torch.no_grad() omitted — JAX is explicit-grad)
    X_test = jnp.array([[4.0], [7.0]])
    predictions = linear_model(params, X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")


if __name__ == "__main__":
    main()