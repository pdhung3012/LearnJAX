import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import pickle


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(key):
    """Generate training data. Returns (X, y).

    PYTORCH EQUIVALENT:
        X = torch.rand(100, 1)
        y = 3 * X + 2 + torch.randn(100, 1) * 0.1

    TRANSLATION NOTES:
        - torch.rand → jax.random.uniform (both produce U[0,1)).
        - torch.randn → jax.random.normal (both produce N(0,1)).
        - JAX requires an explicit PRNGKey; PyTorch consumed from its
          global RNG state.  We split the incoming key so uniform and
          normal draws use independent streams.

    MOCK INJECTION:
        Hardcode X = jnp.array([[0.0], [0.5], [1.0]]) and
        y = jnp.array([[2.0], [3.5], [5.0]]) to test downstream
        functions without randomness.
    """
    key_x, key_noise = random.split(key)
    X = random.uniform(key_x, shape=(100, 1))
    y = 3 * X + 2 + random.normal(key_noise, shape=(100, 1)) * 0.1
    return X, y


# ---------------------------------------------------------------------------
# Model (params dict replaces nn.Module)
# ---------------------------------------------------------------------------

def init_params(key):
    """Initialise model parameters matching PyTorch nn.Linear(1, 1).

    PYTORCH EQUIVALENT:
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(1, 1)

    TRANSLATION NOTES:
        - PyTorch nn.Linear(in=1, out=1) stores weight of shape
          (out_features, in_features) = (1, 1) and bias of shape
          (out_features,) = (1,).  Both are initialised from
          U(-1/sqrt(in_features), 1/sqrt(in_features)) i.e. U(-1, 1)
          (Kaiming uniform).
        - We replicate that distribution here with
          jax.random.uniform(..., minval=-1.0, maxval=1.0).
        - Weight is stored as shape (1, 1) and bias as shape (1,)
          to mirror PyTorch's convention.  The forward pass transposes
          weight before the dot product, exactly as nn.Linear does
          (output = x @ weight.T + bias).

    MOCK INJECTION:
        Hardcode params = {"w": jnp.array([[0.5]]),
                           "b": jnp.array([0.0])} to verify the
        forward pass returns 0.5*x.
    """
    key_w, key_b = random.split(key)
    w = random.uniform(key_w, shape=(1, 1), minval=-1.0, maxval=1.0)
    b = random.uniform(key_b, shape=(1,), minval=-1.0, maxval=1.0)
    return {"w": w, "b": b}


def forward(params, x):
    """Forward pass: x @ w^T + b  (mirrors nn.Linear).

    PYTORCH EQUIVALENT:
        def forward(self, x):
            return self.fc(x)

    TRANSLATION NOTES:
        - nn.Linear computes  output = x @ weight.T + bias.
        - params["w"] has shape (out, in) = (1, 1), so we transpose
          before the dot product to keep shapes identical.

    MOCK INJECTION:
        With params = {"w": jnp.array([[2.0]]), "b": jnp.array([1.0])}
        and x = jnp.array([[3.0]]), expect output jnp.array([[7.0]]).
    """
    return jnp.dot(x, params["w"].T) + params["b"]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def mse_loss(params, X, y):
    """Mean-squared-error loss.

    PYTORCH EQUIVALENT:
        criterion = nn.MSELoss()
        loss = criterion(predictions, y)

    TRANSLATION NOTES:
        - nn.MSELoss() defaults to reduction='mean', so we use
          jnp.mean of squared differences.
        - The forward pass is called inside the loss so that
          jax.value_and_grad can differentiate through it.

    MOCK INJECTION:
        With predictions = jnp.array([[1.0], [2.0]]) and
        y = jnp.array([[1.0], [2.0]]), expect loss = 0.0.
    """
    predictions = forward(params, X)
    return jnp.mean((predictions - y) ** 2)


# ---------------------------------------------------------------------------
# Train step (jitted)
# ---------------------------------------------------------------------------

@jit
def train_step(params, X, y, lr):
    """Single gradient-descent update.

    PYTORCH EQUIVALENT:
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

    TRANSLATION NOTES:
        - optimizer.zero_grad() is unnecessary in JAX because gradients
          are returned fresh from value_and_grad each call.
        - loss.backward() + optimizer.step() is replaced by
          jax.value_and_grad to compute the loss and gradients in one
          pass, followed by a manual SGD update (param -= lr * grad).
        - @jit compiles this function; no Python side effects (print,
          logging) are allowed inside.

    MOCK INJECTION:
        Provide params = {"w": jnp.array([[0.0]]),
                          "b": jnp.array([0.0])},
        X = jnp.array([[1.0]]), y = jnp.array([[5.0]]), lr = 0.1.
        After one step w and b should move towards values that
        reduce MSE from the initial prediction of 0.0 vs target 5.0.
    """
    loss, grads = value_and_grad(mse_loss)(params, X, y)
    params = {k: params[k] - lr * grads[k] for k in params}
    return params, loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(X, y, params, num_epochs, lr=0.01):
    """Run the full training loop, returning updated params.

    PYTORCH EQUIVALENT:
        def train_model(X, y, model, optimizer, criterion, num_epochs):
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                predictions = model(X)
                loss = criterion(predictions, y)
                loss.backward()
                optimizer.step()

    TRANSLATION NOTES:
        - The PyTorch version mutates model weights in-place via
          optimizer.step().  Here we thread the params dict through
          each train_step call, returning the final params.
        - Criterion and optimizer are folded into train_step (loss
          function is fixed as MSE; SGD update is manual with lr).
        - Logging (if any) belongs here, not in train_step, because
          train_step is @jit-compiled.

    MOCK INJECTION:
        Use X = jnp.array([[1.0]]), y = jnp.array([[3.0]]),
        params from init_params, num_epochs = 1, lr = 0.01.
        Verify returned params differ from input and loss is a scalar.
    """
    for epoch in range(num_epochs):
        params, loss = train_step(params, X, y, lr)
    return params


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_params(params, path):
    """Save params dict to disk.

    PYTORCH EQUIVALENT:
        torch.save(model.state_dict(), "model.pth")

    TRANSLATION NOTES:
        - JAX arrays are converted to plain NumPy arrays before
          pickling so the file is framework-agnostic and avoids
          issues with JAX's device-backed arrays.

    MOCK INJECTION:
        Pass params = {"w": jnp.array([[1.0]]), "b": jnp.array([0.0])},
        save, reload, and assert equality.
    """
    numpy_params = {k: v.tolist() for k, v in params.items()}
    with open(path, "wb") as f:
        pickle.dump(numpy_params, f)


def load_params(path):
    """Load params dict from disk.

    PYTORCH EQUIVALENT:
        loaded_model = SimpleModel()
        loaded_model.load_state_dict(torch.load("model.pth"))
        loaded_model.eval()

    TRANSLATION NOTES:
        - We do not need to instantiate a fresh model or call .eval()
          because JAX models are pure functions — the params dict *is*
          the model state.  Loading the dict is sufficient.
        - Values are converted back to jnp arrays from the plain
          Python lists stored by save_params.

    MOCK INJECTION:
        Save known params, load them back, assert all values match
        within floating-point tolerance.
    """
    with open(path, "rb") as f:
        numpy_params = pickle.load(f)
    return {k: jnp.array(v) for k, v in numpy_params.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """End-to-end train → save → load → predict workflow.

    PYTORCH EQUIVALENT:
        def main():
            torch.manual_seed(42)
            model = make_model()
            criterion = make_criterion()
            optimizer = make_optimizer(model)
            X, y = generate_data()
            train_model(...)
            torch.save(...)
            loaded_model = ...
            predictions = loaded_model(X_test)
            print(...)

    TRANSLATION NOTES:
        - torch.manual_seed(42) → jax.random.PRNGKey(42), then
          successive splits mirror the original RNG consumption order:
          first model init, then data generation.
        - make_criterion / make_optimizer have no JAX equivalents as
          separate objects; the loss function is hardcoded in
          mse_loss and SGD is a manual update in train_step.
        - with torch.no_grad() is omitted; JAX only computes
          gradients when explicitly requested via jax.grad.

    MOCK INJECTION:
        Not applicable to main — test individual functions instead.
    """
    # Mirror: torch.manual_seed(42)
    key = random.PRNGKey(42)

    # Mirror original call order: model init first, then data.
    key_model, key_data = random.split(key)
    params = init_params(key_model)
    X, y = generate_data(key_data)

    # Train
    params = train_model(X, y, params, num_epochs=100, lr=0.01)

    # Save and load
    save_params(params, "model.pkl")
    loaded_params = load_params("model.pkl")

    # Predict
    X_test = jnp.array([[0.5], [1.0], [1.5]])
    predictions = forward(loaded_params, X_test)
    print(f"Predictions after loading: {predictions}")


if __name__ == "__main__":
    main()