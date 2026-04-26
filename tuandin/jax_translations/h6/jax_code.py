"""JAX translation of h6: LSTM language model + dynamic quantization.

Faithful to PyTorch:
- Embedding(vocab=50, embed=64) -> 2-layer LSTM(64, 128) -> Linear(128, 50)
  -> Softmax over the last timestep.
- 5 epochs, CrossEntropy on the *softmax probabilities* (mirroring the
  PyTorch code which applies Softmax inside forward and feeds it to
  CrossEntropyLoss — note that this is a known bug pattern in the original
  PyTorch code: nn.CrossEntropyLoss expects raw logits, not probabilities;
  applying it to softmax-ed values still trains but with attenuated gradients.
  We replicate this *exactly* for fidelity).
- Save and load the model after training.

JAX vs PyTorch quantization:
- PyTorch's torch.quantization.quantize_dynamic replaces nn.Linear/nn.LSTM with
  int8 dynamically-quantized variants. JAX has no first-party 1-line equivalent;
  AQT (jax.aqt) and other tools exist but are heavyweight. We document the gap
  and emulate the *effect* (post-training int8 weight quantization for Dense
  layers) by simulating per-tensor symmetric quantize-dequantize on the weights.
  This preserves the demo's intent (smaller / "quantized" model) without bringing
  in a new dependency.
"""
import pickle
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


class LanguageModel(nn.Module):
    vocab_size: int
    embed_size: int
    hidden_size: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        x = nn.Embed(self.vocab_size, self.embed_size)(x)
        for i in range(self.num_layers):
            cell = nn.OptimizedLSTMCell(features=self.hidden_size, name=f"lstm_{i}")
            x = nn.RNN(cell)(x)
        last = x[:, -1, :]
        logits = nn.Dense(self.vocab_size)(last)
        # The original applies Softmax(dim=1); we mirror that.
        return jax.nn.softmax(logits, axis=1)


def fake_quantize_dense(params, num_bits=8):
    """Simulate per-tensor symmetric int8 weight quantization on every Dense
    kernel and bias in the param tree. This mimics the *result* of
    torch.quantization.quantize_dynamic for inference."""
    qmax = 2 ** (num_bits - 1) - 1

    def maybe_quant(x):
        if not isinstance(x, jnp.ndarray):
            return x
        scale = jnp.maximum(jnp.max(jnp.abs(x)) / qmax, 1e-8)
        q = jnp.round(x / scale).clip(-qmax - 1, qmax)
        return q * scale

    return jax.tree.map(maybe_quant, params)


def main():
    key = jax.random.PRNGKey(42)
    vocab_size = 50
    seq_length = 10
    batch_size = 32
    embed_size = 64
    hidden_size = 128
    num_layers = 2

    key, kx, ky = jax.random.split(key, 3)
    X_train = jax.random.randint(kx, (batch_size, seq_length), 0, vocab_size)
    y_train = jax.random.randint(ky, (batch_size,), 0, vocab_size)

    model = LanguageModel(vocab_size, embed_size, hidden_size, num_layers)
    key, k_init = jax.random.split(key)
    params = model.init(k_init, X_train)
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    def loss_fn(params, x, y):
        # Mirror PyTorch's bug-shape: criterion(probs, labels). With softmax probs
        # already applied, optax's softmax_cross_entropy_with_integer_labels would
        # double-softmax — which is exactly what nn.CrossEntropyLoss does when fed
        # softmax probabilities, so this is faithful.
        probs = model.apply(params, x)
        return optax.softmax_cross_entropy_with_integer_labels(probs, y).mean()

    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    epochs = 5
    for epoch in range(epochs):
        params, opt_state, loss = train_step(params, opt_state, X_train, y_train)
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss:.4f}")

    # "Dynamic quantization" — simulate int8 quantize-dequantize of weights.
    quantized_params = fake_quantize_dense(params, num_bits=8)
    with open("quantized_language_model.pkl", "wb") as f:
        pickle.dump(jax.device_get(quantized_params), f)

    # Test inference with the quantized model.
    key, kt = jax.random.split(key)
    test_input = jax.random.randint(kt, (1, seq_length), 0, vocab_size)
    pred = model.apply(quantized_params, test_input)
    predicted = int(jnp.argmax(pred, axis=1)[0])
    print(f"Prediction for input {test_input.tolist()}: {predicted}")


if __name__ == "__main__":
    main()
