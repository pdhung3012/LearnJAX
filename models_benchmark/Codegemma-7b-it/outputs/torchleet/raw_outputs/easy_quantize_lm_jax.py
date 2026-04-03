import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Define a simple Language Model (e.g., an LSTM-based model)
class LanguageModel(nn.Module):
    @nn.compact
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        self.embedding = nn.Embed(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Dense(vocab_size)
        self.softmax = nn.softmax

    def call(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Use the last hidden state for prediction
        output = self.fc(lstm_out[:, -1, :])
        return self.softmax(output)

# Create synthetic training data
vocab_size = 50
seq_length = 10
batch_size = 32
rng = jax.random.PRNGKey(42)
X_train = jax.random.randint(rng, (batch_size, seq_length), 0, vocab_size)
y_train = jax.random.randint(rng, (batch_size,), 0, vocab_size)

# Initialize the model, loss function, and optimizer
embed_size = 64
hidden_size = 128
num_layers = 2
model = LanguageModel(vocab_size, embed_size, hidden_size, num_layers)

criterion = optax.softmax_cross_entropy_with_logits
optimizer = optax.adam(learning_rate=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.init(rng)
    params = model.init(rng, jnp.ones((batch_size, seq_length)))
    opt_state = optimizer.init(params)

    for _ in range(len(X_train)):
        rng, key = jax.random.split(rng)
        x = jax.random.randint(key, (batch_size, seq_length), 0, vocab_size)
        y = jax.random.randint(key, (batch_size,), 0, vocab_size)

        def loss_fn(params):
            logits = model.call(x, params)
            loss = criterion(logits, y)
            return loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        loss, logits = grad_fn(params)
        updates, opt_state = optimizer.update(grad_fn, opt_state, params)
        params = optax.apply_updates(params, updates)

    # Log progress every epoch
    print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss.item():.4f}")

# Now, we will quantize the model dynamically to reduce its size and improve inference speed
# Quantization: Apply dynamic quantization to the language model
quantized_model = jax.jit(jax.tree_util.tree_map(lambda x: x.astype(jnp.qint8), model))

# Save the quantized model
jax.save(quantized_model.state_dict(), "quantized_language_model.ckpt")

# Load the quantized model and test it
quantized_model = LanguageModel(
    vocab_size, embed_size, hidden_size, num_layers)

# Apply dynamic quantization on the model after defining it
quantized_model = jax.jit(jax.tree_util.tree_map(lambda x: x.astype(jnp.qint8), quantized_model))

quantized_model.load_state_dict(jax.load("quantized_language_model.ckpt"))

# Testing the quantized model on a sample input
rng, key = jax.random.split(rng)
test_input = jax.random.randint(key, (1, seq_length), 0, vocab_size)
prediction = quantized_model.call(test_input)
print(
    f"Prediction for input {test_input.tolist()}: {prediction.argmax(axis=1).item()}")