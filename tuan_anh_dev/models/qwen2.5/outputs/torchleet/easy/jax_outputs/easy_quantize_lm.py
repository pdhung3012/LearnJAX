import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import serialization
from flax.training import train_state
import optax
import numpy as np

# Define a simple Language Model (LSTM-based, matching PyTorch: 2-layer LSTM)
class LanguageModel(nn.Module):
    vocab_size: int = 50
    embed_size: int = 64
    hidden_size: int = 128
    num_layers: int = 2

    @nn.compact
    def __call__(self, x):
        # Embedding
        embedded = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_size)(x)

        # Multi-layer LSTM using scan over sequence for each layer
        for layer_i in range(self.num_layers):
            lstm_cell = nn.LSTMCell(features=self.hidden_size, name=f"lstm_{layer_i}")
            batch_size = x.shape[0]
            carry = lstm_cell.initialize_carry(
                jax.random.PRNGKey(0),
                (batch_size, embedded.shape[-1] if layer_i == 0 else self.hidden_size),
            )

            outputs = []
            input_seq = embedded if layer_i == 0 else lstm_out
            for t in range(x.shape[1]):
                carry, h = lstm_cell(carry, input_seq[:, t, :])
                outputs.append(h)
            lstm_out = jnp.stack(outputs, axis=1)

        # Use last hidden state for prediction
        output = nn.Dense(self.vocab_size)(lstm_out[:, -1, :])
        return jax.nn.softmax(output, axis=-1)

# Create synthetic training data
key = jax.random.PRNGKey(42)
vocab_size = 50
seq_length = 10
batch_size = 32
key1, key2 = jax.random.split(key)
X_train = jax.random.randint(key1, (batch_size, seq_length), minval=0, maxval=vocab_size)
y_train = jax.random.randint(key2, (batch_size,), minval=0, maxval=vocab_size)

# Initialize the model, loss function, and optimizer
model = LanguageModel(vocab_size=vocab_size, embed_size=64, hidden_size=128, num_layers=2)
dummy_input = jnp.ones([1, seq_length], dtype=jnp.int32)
variables = model.init(jax.random.PRNGKey(0), dummy_input)

tx = optax.adam(learning_rate=0.001)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=tx
)

# Training loop
epochs = 5
for epoch in range(epochs):
    def loss_fn(params):
        logits = model.apply({'params': params}, X_train)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y_train).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

    print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss.item():.4f}")

# Quantization: JAX equivalent of dynamic quantization
# Restrict quantization to Linear/LSTM-style modules to mirror torch.quantization.quantize_dynamic.
def _is_dynamic_quant_module(path):
    module_name = path.split("/")[0]
    return module_name.startswith("Dense_") or module_name.startswith("lstm_")


def _quantize_array(arr):
    max_abs = np.max(np.abs(arr))
    scale = np.float32(1.0 if max_abs == 0 else max_abs / 127.0)
    q = np.round(arr / scale).astype(np.int8)
    return {"quantized": q, "scale": np.array(scale, dtype=np.float32)}


def quantize_params(params, path=""):
    """Quantize only Dense/LSTM float weights to int8 and store scale factors."""
    quantized = {}
    for k, v in params.items():
        curr_path = f"{path}/{k}" if path else k
        if isinstance(v, dict):
            quantized[k] = quantize_params(v, curr_path)
        else:
            arr = np.asarray(v)
            if arr.dtype in (np.float16, np.float32, np.float64) and _is_dynamic_quant_module(curr_path):
                quantized[k] = _quantize_array(arr)
            else:
                quantized[k] = arr
    return quantized


def dequantize_params(qparams):
    """Dequantize int8 params back to float32."""
    dequantized = {}
    for k, v in qparams.items():
        if isinstance(v, dict) and "quantized" in v and "scale" in v:
            dequantized[k] = jnp.asarray(v["quantized"].astype(np.float32) * np.asarray(v["scale"], dtype=np.float32))
        elif isinstance(v, dict):
            dequantized[k] = dequantize_params(v)
        else:
            dequantized[k] = jnp.asarray(v)
    return dequantized


def save_quantized_params(qparams, filename):
    payload = np.frombuffer(serialization.to_bytes(qparams), dtype=np.uint8)
    np.savez(filename, payload=payload)


def load_quantized_params(filename, template):
    with np.load(filename) as data:
        payload = data["payload"].tobytes()
    return serialization.from_bytes(template, payload)


# Quantize the trained model
quantized_params = quantize_params(jax.tree.map(np.array, state.params))

# Save quantized model
save_quantized_params(quantized_params, "quantized_language_model.npz")

# Load the quantized model and test it
quantized_model = LanguageModel(vocab_size=vocab_size, embed_size=64, hidden_size=128, num_layers=2)
quantized_variables = quantized_model.init(jax.random.PRNGKey(123), dummy_input)
quantized_template = quantize_params(jax.tree.map(np.array, quantized_variables["params"]))
loaded_quantized_params = load_quantized_params("quantized_language_model.npz", quantized_template)

# Dequantize and test
dequantized_params = dequantize_params(loaded_quantized_params)

# Testing the quantized model on a sample input
test_input = jax.random.randint(jax.random.PRNGKey(99), (1, seq_length), minval=0, maxval=vocab_size)
logits = quantized_model.apply({"params": dequantized_params}, test_input)
prediction = jnp.argmax(logits, axis=-1)
print(f"Prediction for input {test_input.tolist()}: {prediction.item()}")
