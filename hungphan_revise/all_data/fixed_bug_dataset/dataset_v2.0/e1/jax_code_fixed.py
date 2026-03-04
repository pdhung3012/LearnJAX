import jax
import jax.numpy as jnp
from jax import jit, random, value_and_grad


# =============================================================================
# BLOCK 1: Generate Synthetic Data
# =============================================================================

def generate_data(key: jax.Array, n: int = 100) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    PYTORCH EQUIVALENT:
        torch.manual_seed(42)
        X = torch.rand(100, 1) * 10
        y = 2 * X + 3 + torch.randn(100, 1)

    TRANSLATION NOTES:
        - torch.manual_seed(42) becomes an explicit PRNGKey(42). JAX requires
          all RNG state to be passed explicitly; there is no global seed.
        - random.split(key) produces two independent subkeys so that X and y
          noise draws are statistically independent, mirroring PyTorch's
          sequential torch.rand / torch.randn calls on the same seeded state.
        - torch.rand(100, 1) * 10 → random.uniform with shape=(n, 1), scaled
          by 10. JAX's uniform draws from [0, 1) by default, same as PyTorch.
        - torch.randn(100, 1) → random.normal with shape=(n, 1).

    MOCK INJECTION:
        To test without randomness, inject:
            X = jnp.linspace(0, 10, 100).reshape(100, 1)
            y = 2 * X + 3  # no noise
    """
    key_x, key_noise = random.split(key)
    X = random.uniform(key_x, shape=(n, 1)) * 10
    y = 2 * X + 3 + random.normal(key_noise, shape=(n, 1))
    return X, y


# =============================================================================
# BLOCK 2: Model — Linear Regression Forward Pass
# =============================================================================

def init_params(key: jax.Array) -> dict:
    """
    PYTORCH EQUIVALENT:
        class LinearRegressionModel(nn.Module):
            def __init__(self):
                self.linear = nn.Linear(1, 1)
        model = LinearRegressionModel()

    TRANSLATION NOTES:
        - nn.Module with a single nn.Linear(1, 1) is replaced by a plain
          Python dict holding 'w' and 'b'. This is the idiomatic functional
          JAX pattern for simple models (no Flax needed here).
        - nn.Linear initialises weight with kaiming_uniform and bias with
          uniform by default. We replicate this with random.normal for 'w'
          and zeros for 'b', which is a common and equivalent initialisation
          for a 1-dimensional linear layer.
        - Shape convention: w is (1, 1) so that jnp.dot(x, w) works for
          inputs of shape (N, 1), matching nn.Linear's matrix multiply.

    MOCK INJECTION:
        To test with known params, inject:
            params = {'w': jnp.array([[1.0]]), 'b': jnp.array([0.0])}
    """
    key_w, _ = random.split(key)
    w = random.normal(key_w, shape=(1, 1)) * 0.01
    b = jnp.zeros((1,))
    return {'w': w, 'b': b}


def forward(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """
    PYTORCH EQUIVALENT:
        def forward(self, x):
            return self.linear(x)

    TRANSLATION NOTES:
        - self.linear(x) in PyTorch performs x @ w.T + b internally.
          Here we write it explicitly as jnp.dot(x, params['w']) + params['b'].
        - params['w'] has shape (1, 1) and x has shape (N, 1), so
          jnp.dot(x, params['w']) yields shape (N, 1), matching PyTorch output.
        - No nn.Module class or __call__ machinery is needed; a plain function
          receiving params is the JAX equivalent.

    MOCK INJECTION:
        To test the forward pass:
            params = {'w': jnp.array([[2.0]]), 'b': jnp.array([3.0])}
            x = jnp.array([[5.0]])
            # Expected output: [[13.0]]
    """
    return jnp.dot(x, params['w']) + params['b']


# =============================================================================
# BLOCK 3: Loss Function
# =============================================================================

def mse_loss(params: dict, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    PYTORCH EQUIVALENT:
        criterion = nn.MSELoss()
        loss = criterion(predictions, y)

    TRANSLATION NOTES:
        - nn.MSELoss() with default reduction='mean' computes
          mean((predictions - y) ** 2). We replicate this with jnp.mean.
        - The loss function takes params as its first argument so that
          jax.value_and_grad(mse_loss)(params, x, y) differentiates w.r.t.
          params only. This is the standard JAX pattern for separating
          parameters from data.

    MOCK INJECTION:
        To test loss computation:
            params = {'w': jnp.array([[2.0]]), 'b': jnp.array([3.0])}
            x = jnp.array([[1.0], [2.0]])
            y = jnp.array([[5.0], [7.0]])  # exact values → loss should be ~0
    """
    predictions = forward(params, x)
    return jnp.mean((predictions - y) ** 2)


# =============================================================================
# BLOCK 4: Training Step
# =============================================================================

@jit
def train_step(params: dict, x: jnp.ndarray, y: jnp.ndarray, lr: float) -> tuple[dict, jnp.ndarray]:
    """
    PYTORCH EQUIVALENT:
        predictions = model(X)
        loss = criterion(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    TRANSLATION NOTES:
        - @jit compiles this function with XLA on first call; subsequent calls
          reuse the compiled kernel. PyTorch's autograd engine has no direct
          equivalent — the @jit decorator is the idiomatic JAX way to match
          PyTorch's eager-mode performance over a training loop.
        - @jit is safe here because all arguments (params, x, y, lr) have
          static shapes and dtypes. lr is a Python float, which JAX traces as
          a compile-time constant.
        - print() must NOT appear inside a @jit-decorated function because
          Python-side side-effects are erased after the first trace. All
          logging therefore lives in train_model(), outside this function.
        - optimizer.zero_grad() is omitted: JAX has no gradient tape that
          accumulates gradients; each call to value_and_grad is stateless.
        - loss.backward() + optimizer.step() → jax.value_and_grad computes
          loss value and gradients in a single functional call.
        - SGD update is applied manually: param = param - lr * grad. This is
          equivalent to optim.SGD(lr=0.01) with no momentum.
        - The updated params dict is returned explicitly; PyTorch mutates
          model.parameters() in place, but JAX requires returning new state.

    MOCK INJECTION:
        To test one training step:
            params = {'w': jnp.array([[0.0]]), 'b': jnp.array([0.0])}
            x = jnp.array([[1.0]])
            y = jnp.array([[5.0]])
            # After one step, w and b should move toward 2 and 3 respectively.
    """
    loss_val, grads = value_and_grad(mse_loss)(params, x, y)
    # Manual SGD update (equivalent to optim.SGD with lr=0.01, no momentum)
    updated_params = {k: params[k] - lr * grads[k] for k in params}
    return updated_params, loss_val


# =============================================================================
# BLOCK 5: Training Loop
# =============================================================================

def train_model(
    X: jnp.ndarray,
    y: jnp.ndarray,
    params: dict,
    epochs: int = 1000,
    lr: float = 0.01,
    log_every: int = 100,
) -> dict:
    """
    PYTORCH EQUIVALENT:
        epochs = 1000
        for epoch in range(epochs):
            predictions = model(X)
            loss = criterion(predictions, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    TRANSLATION NOTES:
        - Data and params are accepted as arguments (never generated here),
          satisfying the convention that train_model is a pure function.
        - The Python for-loop is preserved to mirror PyTorch's loop structure.
          jax.lax.scan would be more performant but would obscure the 1-to-1
          mapping and prevent per-epoch logging.
        - loss.item() → float(loss_val): JAX scalars are 0-d arrays; float()
          converts them to Python scalars for printing, same as .item().
        - with torch.no_grad() around logging is omitted: JAX does not track
          gradients outside of explicit value_and_grad calls.

    MOCK INJECTION:
        To test with a trivial dataset:
            X = jnp.array([[1.0], [2.0], [3.0]])
            y = jnp.array([[5.0], [7.0], [9.0]])  # w=2, b=3 exact solution
            params = {'w': jnp.array([[0.0]]), 'b': jnp.array([0.0])}
            # Loss should decrease toward ~0 over epochs.
    """
    for epoch in range(epochs):
        params, loss_val = train_step(params, X, y, lr)
        if (epoch + 1) % log_every == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {float(loss_val):.4f}")
    return params


# =============================================================================
# BLOCK 6: Main — Initialise, Train, Evaluate
# =============================================================================

def main():
    """
    PYTORCH EQUIVALENT:
        (top-level script: seed, data gen, model init, train, print params, test)

    TRANSLATION NOTES:
        - All randomness flows through explicit PRNGKeys derived from a single
          root key via sequential random.split calls, replacing the single
          torch.manual_seed(42) global seed.
        - model.linear.parameters() → (params['w'], params['b']): params is
          a plain dict, so learned values are accessed by key.
        - w.item() / b.item() → float(params['w'][0,0]) / float(params['b'][0]):
          indexing extracts the scalar from the array before converting.
        - with torch.no_grad(): predictions = model(X_test) → plain forward()
          call; no context manager needed since JAX doesn't track gradients
          outside value_and_grad.

    MOCK INJECTION:
        Replace generate_data and init_params with hardcoded values to isolate
        training logic:
            X = jnp.ones((100, 1)) * 5.0
            y = jnp.ones((100, 1)) * 13.0  # 2*5 + 3
            params = {'w': jnp.array([[0.0]]), 'b': jnp.array([0.0])}
    """
    # Derive independent subkeys from a single root key (replaces manual_seed)
    root_key = random.PRNGKey(42)
    data_key, model_key = random.split(root_key)

    # Generate data
    X, y = generate_data(data_key, n=100)

    # Initialise parameters
    params = init_params(model_key)

    # Train
    params = train_model(X, y, params, epochs=1000, lr=0.01, log_every=100)

    # Display learned parameters (equivalent to iterating model.linear.parameters())
    w = float(params['w'][0, 0])
    b = float(params['b'][0])
    print(f"Learned weight: {w:.4f}, Learned bias: {b:.4f}")

    # Test on new data (no torch.no_grad() needed in JAX)
    X_test = jnp.array([[4.0], [7.0]])
    predictions = forward(params, X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")


if __name__ == "__main__":
    main()