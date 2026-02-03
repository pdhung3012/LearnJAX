import jax
import jax.numpy as jnp
from flax import linen as nn
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
            lstm_cell = nn.LSTMCell(features=self.hidden_size, name=f'lstm_{layer_i}')
            batch_size = x.shape[0]
            carry = lstm_cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, embedded.shape[-1] if layer_i == 0 else self.hidden_size))

            outputs = []
            seq_len = embedded.shape[1] if layer_i == 0 else lstm_out.shape[1]
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
# Quantize model weights to int8 (simulating torch.quantization.quantize_dynamic)
def quantize_params(params):
    """Quantize float32 params to int8 and store scale factors."""
    quantized = {}
    for k, v in params.items():
        if isinstance(v, dict):
            quantized[k] = quantize_params(v)
        else:
            arr = np.array(v)
            if arr.dtype in (np.float32, np.float16):
                scale = np.max(np.abs(arr)) / 127.0
                q = np.round(arr / scale).astype(np.int8)
                quantized[k] = {'quantized': q, 'scale': scale}
            else:
                quantized[k] = v
    return quantized

def dequantize_params(qparams):
    """Dequantize int8 params back to float32."""
    dequantized = {}
    for k, v in qparams.items():
        if isinstance(v, dict) and 'quantized' in v and 'scale' in v:
            dequantized[k] = jnp.array(v['quantized'].astype(np.float32) * v['scale'])
        elif isinstance(v, dict):
            dequantized[k] = dequantize_params(v)
        else:
            dequantized[k] = v
    return dequantized

# Quantize the trained model
quantized_params = quantize_params(jax.tree.map(np.array, state.params))

# Save quantized model
flat_params = jax.tree.leaves(state.params)
save_dict = {f'param_{i}': np.array(p) for i, p in enumerate(flat_params)}
np.savez("quantized_language_model.npz", **save_dict)

# Dequantize and test
dequantized_params = dequantize_params(quantized_params)

# Testing the quantized model on a sample input
test_input = jax.random.randint(jax.random.PRNGKey(99), (1, seq_length), minval=0, maxval=vocab_size)
logits = model.apply({'params': dequantized_params}, test_input)
prediction = jnp.argmax(logits, axis=-1)
print(f"Prediction for input {test_input.tolist()}: {prediction.item()}")
