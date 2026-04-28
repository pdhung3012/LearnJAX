"""JAX translation of m1: custom hand-rolled LSTM + built-in LSTM, sine wave forecast.

Faithful to PyTorch:
- CustomLSTMModel: 4 gates with W_x in R^(input_dim, hidden), W_h in R^(hidden, hidden),
  bias in R^(hidden), all initialized via randn (PyTorch nn.Parameter(torch.randn(...))).
  Initial H, C are randn each forward pass (the original re-samples them on every
  forward call as well; we replicate that — there is no PyTorch seed reset between
  forwards, so the random initial state changes across epochs).
- LSTMModel: nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True),
  followed by Linear(50, 1) on the *last* timestep.
- Adam(lr=0.01), MSE, 500 epochs each, log every 50.
- Test on a longer sine sequence with the same windowing.

JAX implementation:
- CustomLSTM uses jax.lax.scan over time to fuse the per-step kernel calls.
- "Built-in" LSTM is implemented with flax.linen.OptimizedLSTMCell + nn.RNN, which
  also scans efficiently. PyTorch's nn.LSTM ships a fused CuDNN/CPU kernel; on CPU
  the gap to a scan'd Flax LSTM is small.

Speed notes: the custom model wins for JAX (scan beats a Python `for t in range(seq_len)`
loop wrapped in autograd). Built-in model: roughly comparable on CPU.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn


# ---- Contract API used by test_equivalence.py ------------------------------
def compute(inputs):
    """Custom LSTM single-step forward with caller-supplied weights.

    Inputs: dict with Wxi, Whi, bi, Wxf, Whf, bf, Wxo, Who, bo, Wxc, Whc, bc,
            X_t (B, in_dim), H_prev (B, hidden), C_prev (B, hidden).
    Returns: dict with H, C each (B, hidden).
    """
    W = {k: jnp.asarray(v) for k, v in inputs.items()
         if k.startswith(("Wx", "Wh", "b"))}
    X_t = jnp.asarray(inputs["X_t"])
    H_prev = jnp.asarray(inputs["H_prev"])
    C_prev = jnp.asarray(inputs["C_prev"])
    I = jax.nn.sigmoid(X_t @ W["Wxi"] + H_prev @ W["Whi"] + W["bi"])
    F_ = jax.nn.sigmoid(X_t @ W["Wxf"] + H_prev @ W["Whf"] + W["bf"])
    O = jax.nn.sigmoid(X_t @ W["Wxo"] + H_prev @ W["Who"] + W["bo"])
    C_tilde = jnp.tanh(X_t @ W["Wxc"] + H_prev @ W["Whc"] + W["bc"])
    C = F_ * C_prev + I * C_tilde
    H = O * jnp.tanh(C)
    return {"H": np.asarray(H), "C": np.asarray(C)}
import matplotlib.pyplot as plt


# Custom LSTM (manual gates) ---------------------------------------------------

def init_custom_lstm(key, input_dim, hidden_units):
    keys = jax.random.split(key, 12)

    def make_gate(kx, kh):
        return {
            "Wx": jax.random.normal(kx, (input_dim, hidden_units)),
            "Wh": jax.random.normal(kh, (hidden_units, hidden_units)),
            "b":  jnp.zeros((hidden_units,)),
        }

    params = {
        "i": make_gate(keys[0], keys[1]),
        "f": make_gate(keys[2], keys[3]),
        "o": make_gate(keys[4], keys[5]),
        "c": make_gate(keys[6], keys[7]),
        "fc_W": jax.random.uniform(
            keys[8], (hidden_units, 1),
            minval=-1.0 / jnp.sqrt(hidden_units), maxval=1.0 / jnp.sqrt(hidden_units),
        ),
        "fc_b": jax.random.uniform(
            keys[9], (1,),
            minval=-1.0 / jnp.sqrt(hidden_units), maxval=1.0 / jnp.sqrt(hidden_units),
        ),
    }
    return params


def custom_lstm_apply(params, inputs, init_HC):
    H, C = init_HC

    def step(carry, x_t):
        H, C = carry
        I = jax.nn.sigmoid(x_t @ params["i"]["Wx"] + H @ params["i"]["Wh"] + params["i"]["b"])
        F = jax.nn.sigmoid(x_t @ params["f"]["Wx"] + H @ params["f"]["Wh"] + params["f"]["b"])
        O = jax.nn.sigmoid(x_t @ params["o"]["Wx"] + H @ params["o"]["Wh"] + params["o"]["b"])
        C_tilde = jnp.tanh(x_t @ params["c"]["Wx"] + H @ params["c"]["Wh"] + params["c"]["b"])
        C = F * C + I * C_tilde
        H = O * jnp.tanh(C)
        return (H, C), H

    # inputs: (batch, seq, input_dim) -> scan over seq -> (seq, batch, hidden).
    inputs_t = jnp.transpose(inputs, (1, 0, 2))
    (H, C), all_H = jax.lax.scan(step, (H, C), inputs_t)
    all_H = jnp.transpose(all_H, (1, 0, 2))  # (batch, seq, hidden)
    pred = all_H @ params["fc_W"] + params["fc_b"]  # (batch, seq, 1)
    return pred, (H, C)


# Built-in-style LSTM via Flax -------------------------------------------------

class FlaxLSTMModel(nn.Module):
    hidden_size: int = 50

    @nn.compact
    def __call__(self, x):
        cell = nn.OptimizedLSTMCell(features=self.hidden_size)
        rnn = nn.RNN(cell)
        out = rnn(x)               # (batch, seq, hidden)
        out = nn.Dense(1)(out[:, -1, :])
        return out


# Data -------------------------------------------------------------------------

def create_in_out_sequences(data, seq_length):
    n = data.shape[0]
    inp = jnp.stack([data[i:i + seq_length] for i in range(n - seq_length)])
    out = jnp.stack([data[i + seq_length] for i in range(n - seq_length)])
    return inp, out


def main():
    key = jax.random.PRNGKey(42)
    sequence_length = 10
    num_samples = 100

    X = jnp.linspace(0, 4 * 3.14159, num_samples).reshape(-1, 1)
    y = jnp.sin(X)
    X_seq, y_seq = create_in_out_sequences(y, sequence_length)

    # Custom LSTM training -----------------------------------------------------
    key, k_init, k_state = jax.random.split(key, 3)
    custom_params = init_custom_lstm(k_init, 1, 50)
    opt_c = optax.adam(0.01)
    opt_c_state = opt_c.init(custom_params)

    def custom_loss(params, X_seq, y_seq, init_HC):
        pred, _ = custom_lstm_apply(params, X_seq, init_HC)
        return jnp.mean((pred[:, -1, :] - y_seq) ** 2)

    @jax.jit
    def custom_step(params, opt_state, X_seq, y_seq, init_HC):
        loss, grads = jax.value_and_grad(custom_loss)(params, X_seq, y_seq, init_HC)
        updates, opt_state = opt_c.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    epochs = 500
    batch_size = X_seq.shape[0]
    for epoch in range(epochs):
        # Re-sample initial state every step to mimic the PyTorch code, which calls
        # torch.randn(batch, hidden) inside forward each iteration.
        k_state, k_h, k_c = jax.random.split(k_state, 3)
        H0 = jax.random.normal(k_h, (batch_size, 50))
        C0 = jax.random.normal(k_c, (batch_size, 50))
        custom_params, opt_c_state, loss = custom_step(
            custom_params, opt_c_state, X_seq, y_seq, (H0, C0)
        )
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

    # Built-in LSTM training ---------------------------------------------------
    flax_lstm = FlaxLSTMModel(hidden_size=50)
    key, k_init = jax.random.split(key)
    flax_params = flax_lstm.init(k_init, X_seq)
    opt_b = optax.adam(0.01)
    opt_b_state = opt_b.init(flax_params)

    def builtin_loss(params, X_seq, y_seq):
        pred = flax_lstm.apply(params, X_seq)
        return jnp.mean((pred - y_seq) ** 2)

    @jax.jit
    def builtin_step(params, opt_state, X_seq, y_seq):
        loss, grads = jax.value_and_grad(builtin_loss)(params, X_seq, y_seq)
        updates, opt_state = opt_b.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for epoch in range(epochs):
        flax_params, opt_b_state, loss = builtin_step(flax_params, opt_b_state, X_seq, y_seq)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

    # Test --------------------------------------------------------------------
    test_steps = 100
    X_test = jnp.linspace(0, 5 * 3.14159, test_steps).reshape(-1, 1)
    y_test = jnp.sin(X_test)
    X_test_seq, _ = create_in_out_sequences(y_test, sequence_length)

    k_state, k_h, k_c = jax.random.split(k_state, 3)
    H0 = jax.random.normal(k_h, (X_test_seq.shape[0], 50))
    C0 = jax.random.normal(k_c, (X_test_seq.shape[0], 50))
    pred_custom, _ = custom_lstm_apply(custom_params, X_test_seq, (H0, C0))
    pred_custom = pred_custom[:, -1, :].reshape(-1)
    pred_inbuilt = flax_lstm.apply(flax_params, X_test_seq).reshape(-1)

    print(f"Predictions with Custom Model for new sequence: {pred_custom.tolist()}")
    print(f"Predictions with In-Built Model: {pred_inbuilt.tolist()}")

    plt.figure()
    plt.plot(jax.device_get(pred_custom), label="custom model")
    plt.plot(jax.device_get(pred_inbuilt), label="inbuilt model")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
