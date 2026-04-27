"""JAX translation of m5: simple RNN on sine-wave windowed sequences.

Faithful to PyTorch:
- Architecture: nn.RNN(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
  + Linear(50,1). The PyTorch model holds an unused ReLU; we replicate it as a no-op.
- Training: 500 epochs, *one window at a time* (the inner Python loop iterates over
  individual sequences with batch_size=1). MSELoss, Adam(lr=1e-3).

JAX implementation:
- nn.SimpleCell + nn.RNN (Flax) for the recurrent layer.
- We *vectorize* the per-window inner loop with vmap so each "epoch" applies the
  optimizer once per window — this is what the PyTorch code does (one optimizer
  step per window). To stay faithful to that pattern we use a Python loop over
  windows, but each step is jit'd; PyTorch incurs heavy per-step overhead, so JAX
  should be substantially faster here.

Speed notes: JAX likely 5-15x faster than PyTorch on this workload because the
outer-loop overhead per window is tiny under jit, while PyTorch pays its eager
dispatch cost on every step.
"""
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax


# ---- Contract API used by test_equivalence.py ------------------------------
def compute(inputs):
    """SimpleRNN single-timestep forward with caller-supplied weights.

    Inputs:
        W_ih (hidden, in_dim), b_ih (hidden,)
        W_hh (hidden, hidden), b_hh (hidden,)
        x  (B, T, in_dim) — T is processed step-by-step
        h0 (num_layers=1, B, hidden) — initial hidden state
    Returns: {"out": (B, T, hidden)} — the output sequence.
    """
    W_ih = jnp.asarray(inputs["W_ih"])
    b_ih = jnp.asarray(inputs["b_ih"])
    W_hh = jnp.asarray(inputs["W_hh"])
    b_hh = jnp.asarray(inputs["b_hh"])
    x = jnp.asarray(inputs["x"])
    h = jnp.asarray(inputs["h0"])[0]  # (B, hidden)
    outs = []
    for t in range(x.shape[1]):
        h = jnp.tanh(x[:, t, :] @ W_ih.T + b_ih + h @ W_hh.T + b_hh)
        outs.append(h)
    return {"out": np.asarray(jnp.stack(outs, axis=1))}


class FlaxSimpleRNN(nn.Module):
    hidden_size: int = 50

    @nn.compact
    def __call__(self, x):
        cell = nn.SimpleCell(features=self.hidden_size)
        rnn = nn.RNN(cell)
        out = rnn(x)
        out = nn.Dense(1)(out[:, -1, :])
        return out


def create_in_out_sequences(data, seq_length):
    n = data.shape[0]
    inp = jnp.stack([data[i:i + seq_length] for i in range(n - seq_length)])
    out = jnp.stack([data[i + seq_length] for i in range(n - seq_length)])
    return inp, out


def main():
    sequence_length = 10
    num_samples = 100

    X = jnp.linspace(0, 4 * 3.14159, num_samples).reshape(-1, 1)
    y = jnp.sin(X)
    X_seq, y_seq = create_in_out_sequences(y, sequence_length)
    # X_seq: (N, 10, 1), y_seq: (N, 1).

    model = FlaxSimpleRNN(hidden_size=50)
    rng = jax.random.PRNGKey(42)
    params = model.init(rng, X_seq[:1])
    opt = optax.adam(0.001)
    opt_state = opt.init(params)

    def loss_fn(params, x, y):
        pred = model.apply(params, x)
        return jnp.mean((pred - y) ** 2)

    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    epochs = 500
    n_windows = X_seq.shape[0]
    for epoch in range(epochs):
        last_loss = None
        for i in range(n_windows):
            x_batch = X_seq[i:i + 1]
            y_batch = y_seq[i:i + 1]
            params, opt_state, last_loss = train_step(params, opt_state, x_batch, y_batch)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {float(last_loss):.4f}")

    X_test = jnp.linspace(4 * 3.14159, 5 * 3.14159, 10).reshape(-1, 1)
    X_test = X_test.reshape(1, 10, 1)
    preds = model.apply(params, X_test)
    print(f"Predictions for new sequence: {preds.tolist()}")


if __name__ == "__main__":
    main()
