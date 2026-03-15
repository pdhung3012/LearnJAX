"""
Full JAX translation of the provided PyTorch script.

Fixes applied:
1) JIT static-arg issue: optimizer + criterion are closed over as module globals.
2) grad int32 issue: params dict contains ONLY floating arrays (no int metadata fields).

Structure mirrors PyTorch 1-to-1:
- Data utils
- Data generation
- Models
- Factories
- Training
- Main
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random
import optax
import matplotlib.pyplot as plt

Array = jnp.ndarray
Params = Dict[str, Any]
OptState = optax.OptState


# -----------------------------
# Data utilities (importable)
# -----------------------------
def create_in_out_sequences(data: Array, seq_length: int) -> Tuple[Array, Array]:
    """
    PYTORCH EQUIVALENT:
        def create_in_out_sequences(data, seq_length):
            in_seq = []
            out_seq = []
            for i in range(len(data) - seq_length):
                in_seq.append(data[i:i + seq_length])
                out_seq.append(data[i + seq_length])
            return torch.stack(in_seq), torch.stack(out_seq)

    TRANSLATION NOTES:
        - Python loop + list accumulation + jnp.stack to mirror PyTorch control flow.
        - Not jitted; uses len(data) exactly like PyTorch len(tensor).

    MOCK INJECTION:
        - data=jnp.arange(0, 12, dtype=jnp.float32).reshape(-1, 1), seq_length=10
        - Expect X_seq shape (2, 10, 1), y_seq shape (2, 1)
    """
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i : i + seq_length])
        out_seq.append(data[i + seq_length])
    return jnp.stack(in_seq, axis=0), jnp.stack(out_seq, axis=0)


def generate_data() -> Tuple[Array, Array]:
    """
    PYTORCH EQUIVALENT:
        def generate_data():
            sequence_length = 10
            num_samples = 100
            X = torch.linspace(0, 4 * 3.14159, steps=num_samples).unsqueeze(1)
            y = torch.sin(X)
            X_seq, y_seq = create_in_out_sequences(y, sequence_length)
            return X_seq, y_seq

    TRANSLATION NOTES:
        - jnp.linspace + reshape(-1,1) mirrors torch.linspace + unsqueeze(1).
        - No RNG here (same as PyTorch).

    MOCK INJECTION:
        - Temporarily set num_samples=12, sequence_length=3 to verify shapes.
    """
    sequence_length = 10
    num_samples = 100

    X = jnp.linspace(0.0, 4.0 * 3.14159, num=num_samples).reshape(-1, 1)
    y = jnp.sin(X)

    X_seq, y_seq = create_in_out_sequences(y, sequence_length)
    return X_seq, y_seq


# -----------------------------
# Models (module-level, importable)
# -----------------------------
def init_custom_lstm_params(key: Array, input_dim: int, hidden_units: int) -> Params:
    """
    PYTORCH EQUIVALENT:
        class CustomLSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_units):
                weights_biases_init = lambda: (
                    nn.Parameter(torch.randn(input_dim, hidden_units)),
                    nn.Parameter(torch.randn(hidden_units, hidden_units)),
                    nn.Parameter(torch.zeros(hidden_units)),
                )
                ...
                self.fc = nn.Linear(hidden_units, 1)

    TRANSLATION NOTES:
        - Gate matrices match exactly: randn for W_x and W_h, zeros for bias.
        - For nn.Linear(hidden_units, 1), PyTorch uses uniform init in reset_parameters.
          We emulate that: U(-bound, bound), bound=1/sqrt(in_features) for both weights and bias.
        - IMPORTANT: params dict contains ONLY float arrays (no int metadata), to avoid grad(int32) errors.

    MOCK INJECTION:
        - key=PRNGKey(0), input_dim=1, hidden_units=2 -> check shapes
    """
    def weights_biases_init(k: Array) -> Tuple[Array, Array, Array, Array]:
        k1, k2, k3, k_next = random.split(k, 4)
        W_x = random.normal(k1, (input_dim, hidden_units))
        W_h = random.normal(k2, (hidden_units, hidden_units))
        b = jnp.zeros((hidden_units,), dtype=jnp.float32)
        _ = k3
        return W_x, W_h, b, k_next

    def linear_init(k: Array, in_features: int, out_features: int) -> Tuple[Array, Array]:
        bound = 1.0 / jnp.sqrt(jnp.array(in_features, dtype=jnp.float32))
        k_w, k_b = random.split(k, 2)
        w = random.uniform(k_w, (in_features, out_features), minval=-bound, maxval=bound)
        b = random.uniform(k_b, (out_features,), minval=-bound, maxval=bound)
        return w, b

    Wxi, Whi, bi, key = weights_biases_init(key)
    Wxf, Whf, bf, key = weights_biases_init(key)
    Wxo, Who, bo, key = weights_biases_init(key)
    Wxc, Whc, bc, key = weights_biases_init(key)

    fc_w, fc_b = linear_init(key, hidden_units, 1)

    return {
        "Wxi": Wxi, "Whi": Whi, "bi": bi,
        "Wxf": Wxf, "Whf": Whf, "bf": bf,
        "Wxo": Wxo, "Who": Who, "bo": bo,
        "Wxc": Wxc, "Whc": Whc, "bc": bc,
        "fc_w": fc_w, "fc_b": fc_b,
    }


def custom_lstm_forward(
    params: Params,
    inputs: Array,
    rng_key: Array,
    H_C: Optional[Tuple[Array, Array]] = None,
) -> Tuple[Array, Tuple[Array, Array], Array]:
    """
    PYTORCH EQUIVALENT:
        def forward(self, inputs, H_C=None):
            batch_size, seq_len, _ = inputs.shape
            if not H_C:
                H = torch.randn(batch_size, self.hidden_units)
                C = torch.randn(batch_size, self.hidden_units)
            else:
                H, C = H_C
            ... loop over t ...
            pred = fc(outputs)
            return pred, (H, C)

    TRANSLATION NOTES:
        - Keeps RNG inside forward where PyTorch uses torch.randn.
        - Removes dependence on integer metadata in params by inferring hidden_units from weight shapes.
        - List append + concatenate matches PyTorch cat.

    MOCK INJECTION:
        - inputs=jnp.ones((3,4,1)), rng_key=PRNGKey(0)
        - pred shape should be (3,4,1)
    """
    batch_size, seq_len, _ = inputs.shape
    hidden_units = params["Whi"].shape[0]  # inferred, avoids int leaf in params

    if not H_C:
        kH, kC, rng_key = random.split(rng_key, 3)
        H = random.normal(kH, (batch_size, hidden_units))
        C = random.normal(kC, (batch_size, hidden_units))
    else:
        H, C = H_C

    all_hidden_states = []
    for t in range(seq_len):
        X_t = inputs[:, t, :]

        I_t = jax.nn.sigmoid(jnp.matmul(X_t, params["Wxi"]) + jnp.matmul(H, params["Whi"]) + params["bi"])
        F_t = jax.nn.sigmoid(jnp.matmul(X_t, params["Wxf"]) + jnp.matmul(H, params["Whf"]) + params["bf"])
        O_t = jax.nn.sigmoid(jnp.matmul(X_t, params["Wxo"]) + jnp.matmul(H, params["Who"]) + params["bo"])
        C_tilde = jnp.tanh(jnp.matmul(X_t, params["Wxc"]) + jnp.matmul(H, params["Whc"]) + params["bc"])

        C = F_t * C + I_t * C_tilde
        H = O_t * jnp.tanh(C)

        all_hidden_states.append(H[:, None, :])

    outputs = jnp.concatenate(all_hidden_states, axis=1)
    pred = jnp.matmul(outputs, params["fc_w"]) + params["fc_b"]
    return pred, (H, C), rng_key


def init_inbuilt_lstm_params(key: Array, input_size: int = 1, hidden_size: int = 50) -> Params:
    """
    PYTORCH EQUIVALENT:
        class LSTMModel(nn.Module):
            def __init__(self):
                self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
                self.fc = nn.Linear(50, 1)

    TRANSLATION NOTES:
        - Implements single-layer LSTM with standard parameterization.
        - Uses uniform init bound=1/sqrt(hidden_size) (close to PyTorch default).
        - IMPORTANT: params dict contains ONLY float arrays (no int metadata).

    MOCK INJECTION:
        - hidden_size=2 -> W_x (1,8), W_h (2,8), b (8,)
    """
    bound = 1.0 / jnp.sqrt(jnp.array(hidden_size, dtype=jnp.float32))
    k1, k2, k3, k4, k5 = random.split(key, 5)

    W_x = random.uniform(k1, (input_size, 4 * hidden_size), minval=-bound, maxval=bound)
    W_h = random.uniform(k2, (hidden_size, 4 * hidden_size), minval=-bound, maxval=bound)
    b = random.uniform(k3, (4 * hidden_size,), minval=-bound, maxval=bound)

    fc_bound = 1.0 / jnp.sqrt(jnp.array(hidden_size, dtype=jnp.float32))
    fc_w = random.uniform(k4, (hidden_size, 1), minval=-fc_bound, maxval=fc_bound)
    fc_b = random.uniform(k5, (1,), minval=-fc_bound, maxval=fc_bound)

    return {"W_x": W_x, "W_h": W_h, "b": b, "fc_w": fc_w, "fc_b": fc_b}


def inbuilt_lstm_forward(params: Params, x: Array) -> Array:
    """
    PYTORCH EQUIVALENT:
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

    TRANSLATION NOTES:
        - Explicit time loop; returns final hidden passed through fc.
        - Hidden size inferred from parameter shapes (no int metadata).

    MOCK INJECTION:
        - x=jnp.ones((3,4,1)), params with hidden_size=2 -> output (3,1)
    """
    B, T, _ = x.shape
    hidden_size = params["W_h"].shape[0]

    H = jnp.zeros((B, hidden_size), dtype=jnp.float32)
    C = jnp.zeros((B, hidden_size), dtype=jnp.float32)

    W_x = params["W_x"]
    W_h = params["W_h"]
    b = params["b"]

    for t in range(T):
        X_t = x[:, t, :]
        gates = jnp.matmul(X_t, W_x) + jnp.matmul(H, W_h) + b
        i, f, g, o = jnp.split(gates, 4, axis=-1)

        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        o = jax.nn.sigmoid(o)
        g = jnp.tanh(g)

        C = f * C + i * g
        H = o * jnp.tanh(C)

    out = jnp.matmul(H, params["fc_w"]) + params["fc_b"]
    return out


# -----------------------------
# Factories (importable)
# -----------------------------
def make_model_params(rng_key: Array) -> Tuple[Params, Params, Array]:
    """
    PYTORCH EQUIVALENT:
        def make_model():
            model_custom = CustomLSTMModel(1, 50)
            model_inbuilt = LSTMModel()
            return model_custom, model_inbuilt

    TRANSLATION NOTES:
        - Splits rng_key to initialize both models deterministically.
        - Returns updated rng_key.

    MOCK INJECTION:
        - rng_key=PRNGKey(42)
    """
    k1, k2, rng_key = random.split(rng_key, 3)
    params_custom = init_custom_lstm_params(k1, input_dim=1, hidden_units=50)
    params_inbuilt = init_inbuilt_lstm_params(k2, input_size=1, hidden_size=50)
    return params_custom, params_inbuilt, rng_key


def make_criterion():
    """
    PYTORCH EQUIVALENT:
        def make_criterion():
            return nn.MSELoss()

    TRANSLATION NOTES:
        - Mean squared error with mean reduction.

    MOCK INJECTION:
        - pred=jnp.array([1.,2.]), y=jnp.array([0.,2.]) -> 0.5
    """
    def mse(pred: Array, target: Array) -> Array:
        return jnp.mean((pred - target) ** 2)
    return mse


def make_optimizer() -> optax.GradientTransformation:
    """
    PYTORCH EQUIVALENT:
        def make_optimizer(model):
            return optim.Adam(model.parameters(), lr=0.01)

    TRANSLATION NOTES:
        - optax.adam(lr=0.01)

    MOCK INJECTION:
        - opt=make_optimizer(); opt.init(params)
    """
    return optax.adam(learning_rate=0.01)


# -----------------------------
# Training (importable)
# -----------------------------
CRITERION_FN = make_criterion()
OPTIMIZER = make_optimizer()


@jax.jit
def train_step_custom(
    X: Array,
    y: Array,
    params: Params,
    opt_state: OptState,
    rng_key: Array,
) -> Tuple[Params, OptState, Array, Array]:
    """
    PYTORCH EQUIVALENT:
        pred, state = model_custom(X_seq, state)   # state=None each epoch
        loss = criterion(pred[:, -1, :], y_seq)
        optimizer_custom.zero_grad()
        loss.backward()
        optimizer_custom.step()

    TRANSLATION NOTES:
        - Only array/pytree args are passed into @jit (fixes static arg issue).
        - criterion + optimizer are closed over as globals.
        - RNG threaded explicitly because PyTorch forward samples H,C each epoch.

    MOCK INJECTION:
        - X=jnp.ones((2,3,1)), y=jnp.ones((2,1))
        - rng_key=PRNGKey(1), params from init_custom_lstm_params(PRNGKey(0),1,2)
    """
    def loss_fn(p: Params, key: Array):
        pred, _state, key2 = custom_lstm_forward(p, X, key, H_C=None)
        loss = CRITERION_FN(pred[:, -1, :], y)
        return loss, key2

    (loss, rng_key2), grads = jax.value_and_grad(lambda p: loss_fn(p, rng_key), has_aux=True)(params)

    updates, opt_state2 = OPTIMIZER.update(grads, opt_state, params)
    params2 = optax.apply_updates(params, updates)
    return params2, opt_state2, loss, rng_key2


@jax.jit
def train_step_inbuilt(
    X: Array,
    y: Array,
    params: Params,
    opt_state: OptState,
) -> Tuple[Params, OptState, Array]:
    """
    PYTORCH EQUIVALENT:
        pred = model_inbuilt(X_seq)
        loss = criterion(pred, y_seq)
        optimizer_inbuilt.zero_grad()
        loss.backward()
        optimizer_inbuilt.step()

    TRANSLATION NOTES:
        - Only array/pytree args in @jit.
        - criterion + optimizer are globals.

    MOCK INJECTION:
        - X=jnp.ones((2,3,1)), y=jnp.ones((2,1))
        - params from init_inbuilt_lstm_params(PRNGKey(0),1,2)
    """
    def loss_fn(p: Params):
        pred = inbuilt_lstm_forward(p, X)
        return CRITERION_FN(pred, y)

    loss, grads = jax.value_and_grad(loss_fn)(params)

    updates, opt_state2 = OPTIMIZER.update(grads, opt_state, params)
    params2 = optax.apply_updates(params, updates)
    return params2, opt_state2, loss


def train_model(
    X: Array,
    y: Array,
    params: Params,
    opt_state: OptState,
    num_epochs: int,
    rng_key: Array,
) -> Tuple[Params, OptState, Array]:
    """
    PYTORCH EQUIVALENT:
        def train_model(...):
            for epoch in range(num_epochs):
                state = None
                pred, state = model(X, state)
                loss = criterion(pred[:, -1, :], y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (epoch + 1) % 50 == 0:
                    print(...)

    TRANSLATION NOTES:
        - Logging outside jit.
        - Accepts (X,y,params,...) only.

    MOCK INJECTION:
        - epochs=2 with small tensors.
    """
    loss = jnp.array(0.0, dtype=jnp.float32)
    for epoch in range(num_epochs):
        params, opt_state, loss, rng_key = train_step_custom(X, y, params, opt_state, rng_key)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {float(loss):.4f}")
    return params, opt_state, rng_key


def train_inbuilt_model(
    X: Array,
    y: Array,
    params: Params,
    opt_state: OptState,
    num_epochs: int,
) -> Tuple[Params, OptState]:
    """
    PYTORCH EQUIVALENT:
        def train_inbuilt_model(...):
            for epoch in range(num_epochs):
                pred = model(X)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (epoch + 1) % 50 == 0:
                    print(...)

    TRANSLATION NOTES:
        - Logging outside jit.
        - Accepts (X,y,params,...) only.

    MOCK INJECTION:
        - epochs=2 with small tensors.
    """
    loss = jnp.array(0.0, dtype=jnp.float32)
    for epoch in range(num_epochs):
        params, opt_state, loss = train_step_inbuilt(X, y, params, opt_state)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {float(loss):.4f}")
    return params, opt_state


# -----------------------------
# Main (identical prints/flow)
# -----------------------------
def main():
    """
    PYTORCH EQUIVALENT:
        torch.manual_seed(42)
        X_seq, y_seq = generate_data()
        model_custom, model_inbuilt = make_model()
        criterion = make_criterion()
        optimizer_custom = make_optimizer(model_custom)
        optimizer_inbuilt = make_optimizer(model_inbuilt)
        epochs = 500
        train_model(...)
        epochs = 500
        train_inbuilt_model(...)
        test data -> predictions -> print -> plot

    TRANSLATION NOTES:
        - torch.manual_seed(42) -> rng_key = PRNGKey(42)
        - Call order mirrors PyTorch.

    MOCK INJECTION:
        - For quick smoke test: set epochs=2, test_steps=12
    """
    rng_key = random.PRNGKey(42)

    X_seq, y_seq = generate_data()

    params_custom, params_inbuilt, rng_key = make_model_params(rng_key)

    opt_state_custom = OPTIMIZER.init(params_custom)
    opt_state_inbuilt = OPTIMIZER.init(params_inbuilt)

    epochs = 500
    params_custom, opt_state_custom, rng_key = train_model(
        X_seq, y_seq, params_custom, opt_state_custom, epochs, rng_key
    )

    epochs = 500
    params_inbuilt, opt_state_inbuilt = train_inbuilt_model(
        X_seq, y_seq, params_inbuilt, opt_state_inbuilt, epochs
    )

    # Testing on new data (same logic as original)
    sequence_length = 10
    test_steps = 100

    X_test = jnp.linspace(0.0, 5.0 * 3.14159, num=test_steps).reshape(-1, 1)
    y_test = jnp.sin(X_test)

    X_test_seq, _ = create_in_out_sequences(y_test, sequence_length)

    pred_custom, _state, rng_key = custom_lstm_forward(params_custom, X_test_seq, rng_key, H_C=None)
    pred_inbuilt = inbuilt_lstm_forward(params_inbuilt, X_test_seq)

    pred_custom = jnp.ravel(pred_custom[:, -1, :])
    pred_inbuilt = jnp.squeeze(pred_inbuilt)

    print(f"Predictions with Custom Model for new sequence: {pred_custom.tolist()}")
    print(f"Predictions with In-Built Model: {pred_inbuilt.tolist()}")

    plt.figure()
    plt.plot(pred_custom, label="custom model")
    plt.plot(pred_inbuilt, label="inbuilt model")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()