import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
import optax
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Data Preparation
# ---------------------------
# FIXED: Use same seed as PyTorch (42)
sequence_length = 10
num_samples = 100

# Create a sine wave dataset
X = jnp.linspace(0, 4 * np.pi, num_samples).reshape(-1, 1)
y = jnp.sin(X)

def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i:i+seq_length])
        out_seq.append(data[i+seq_length])
    return jnp.stack(in_seq), jnp.stack(out_seq)

X_seq, y_seq = create_in_out_sequences(y, sequence_length)

# ---------------------------
# Custom LSTM Model
# ---------------------------
class CustomLSTMModel(nn.Module):
    input_dim: int
    hidden_units: int

    def setup(self):
        # FIXED: Use std=1.0 to match PyTorch's torch.randn()
        # PyTorch uses torch.randn() which has mean=0, std=1.0
        # JAX nn.initializers.normal() defaults to std=0.01, so we override
        def weights_biases_init(prefix):
            W = self.param(prefix + "_W", nn.initializers.normal(stddev=1.0), 
                          (self.input_dim, self.hidden_units))
            U = self.param(prefix + "_U", nn.initializers.normal(stddev=1.0), 
                          (self.hidden_units, self.hidden_units))
            b = self.param(prefix + "_b", nn.initializers.zeros, (self.hidden_units,))
            return W, U, b
        
        # Initialize parameters for each gate
        self.Wxi, self.Whi, self.bi = weights_biases_init("input")
        self.Wxf, self.Whf, self.bf = weights_biases_init("forget")
        self.Wxo, self.Who, self.bo = weights_biases_init("output")
        self.Wxc, self.Whc, self.bc = weights_biases_init("candidate")
        self.fc = nn.Dense(features=1)

    def __call__(self, inputs, state=None, use_running_rng=False):
        batch_size, seq_len, _ = inputs.shape
        
        # FIXED: Match PyTorch's "if not H_C" logic
        # PyTorch checks falsy, JAX should check None
        if state is None:
            if use_running_rng:
                # Use RNG for initialization (needed for deterministic behavior)
                H = jax.random.normal(self.make_rng("lstm"), (batch_size, self.hidden_units))
                C = jax.random.normal(self.make_rng("lstm"), (batch_size, self.hidden_units))
            else:
                # Fallback to zeros (shouldn't happen if RNG provided)
                H = jnp.zeros((batch_size, self.hidden_units))
                C = jnp.zeros((batch_size, self.hidden_units))
        else:
            H, C = state

        all_hidden_states = []
        for t in range(seq_len):
            X_t = inputs[:, t, :]
            I_t = jax.nn.sigmoid(jnp.dot(X_t, self.Wxi) + jnp.dot(H, self.Whi) + self.bi)
            F_t = jax.nn.sigmoid(jnp.dot(X_t, self.Wxf) + jnp.dot(H, self.Whf) + self.bf)
            O_t = jax.nn.sigmoid(jnp.dot(X_t, self.Wxo) + jnp.dot(H, self.Who) + self.bo)
            C_tilde = jnp.tanh(jnp.dot(X_t, self.Wxc) + jnp.dot(H, self.Whc) + self.bc)
            C = F_t * C + I_t * C_tilde
            H = O_t * jnp.tanh(C)
            all_hidden_states.append(H[:, None, :])
        
        outputs = jnp.concatenate(all_hidden_states, axis=1)
        pred = self.fc(outputs)
        return pred, (H, C)

# ---------------------------
# In-Built LSTM Model using Flax's LSTMCell
# ---------------------------
class LSTMModel(nn.Module):
    hidden_size: int = 50

    def setup(self):
        self.lstm = nn.LSTMCell(features=self.hidden_size)
        self.fc = nn.Dense(features=1)

    def __call__(self, inputs):
        batch_size, seq_len, _ = inputs.shape
        input_shape = (batch_size,)
        carry = self.lstm.initialize_carry(self.make_rng("lstm"), input_shape)
        outputs = []
        for t in range(seq_len):
            carry, out = self.lstm(carry, inputs[:, t, :])
            outputs.append(out)
        outputs = jnp.stack(outputs, axis=1)
        out = self.fc(outputs[:, -1, :])
        return out

# ---------------------------
# Training: Custom LSTM Model
# ---------------------------
custom_model = CustomLSTMModel(input_dim=1, hidden_units=50)

# FIXED: Use seed 42 to match PyTorch
rng_custom = random.PRNGKey(42)
params_custom = custom_model.init({"params": rng_custom, "lstm": rng_custom}, X_seq)["params"]
optimizer_custom = optax.adam(0.01)
opt_state_custom = optimizer_custom.init(params_custom)

def loss_fn_custom(params, X_seq, y_seq, rng):
    # FIXED: Pass state=None to match PyTorch's state reset behavior
    pred, _ = custom_model.apply(
        {"params": params}, 
        X_seq, 
        state=None,  # Reset state like PyTorch
        use_running_rng=True,
        rngs={"lstm": rng}
    )
    loss = jnp.mean((pred[:, -1, :] - y_seq) ** 2)
    return loss

epochs = 500
for epoch in range(epochs):
    rng_custom, subkey = random.split(rng_custom)
    loss_value, grads = jax.value_and_grad(loss_fn_custom)(params_custom, X_seq, y_seq, subkey)
    updates, opt_state_custom = optimizer_custom.update(grads, opt_state_custom)
    params_custom = optax.apply_updates(params_custom, updates)
    if (epoch + 1) % 50 == 0:
        # FIXED: Match PyTorch print format (no "Custom Model -" prefix)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss_value:.4f}")

# ---------------------------
# Training: In-Built LSTM Model
# ---------------------------
inbuilt_model = LSTMModel(hidden_size=50)

# FIXED: Use same seed 42 as custom model (matching PyTorch)
rng_inbuilt = random.PRNGKey(42)
params_inbuilt = inbuilt_model.init({"params": rng_inbuilt, "lstm": rng_inbuilt}, X_seq)["params"]
optimizer_inbuilt = optax.adam(0.01)
opt_state_inbuilt = optimizer_inbuilt.init(params_inbuilt)

def loss_fn_inbuilt(params, X_seq, y_seq, rng):
    pred = inbuilt_model.apply({"params": params}, X_seq, rngs={"lstm": rng})
    loss = jnp.mean((pred - y_seq) ** 2)
    return loss

for epoch in range(epochs):
    rng_inbuilt, subkey = random.split(rng_inbuilt)
    loss_value, grads = jax.value_and_grad(loss_fn_inbuilt)(params_inbuilt, X_seq, y_seq, subkey)
    updates, opt_state_inbuilt = optimizer_inbuilt.update(grads, opt_state_inbuilt)
    params_inbuilt = optax.apply_updates(params_inbuilt, updates)
    if (epoch + 1) % 50 == 0:
        # FIXED: Match PyTorch print format
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss_value:.4f}")

# ---------------------------
# Testing on New Data
# ---------------------------
test_steps = 100
X_test = jnp.linspace(0, 5 * np.pi, test_steps).reshape(-1, 1)
y_test = jnp.sin(X_test)
X_test_seq, _ = create_in_out_sequences(y_test, sequence_length)

# Get predictions from both models
rng_custom, subkey = random.split(rng_custom)
pred_custom, _ = custom_model.apply(
    {"params": params_custom}, 
    X_test_seq, 
    state=None,
    use_running_rng=True,
    rngs={"lstm": subkey}
)
pred_custom = jnp.squeeze(pred_custom[:, -1, :])

rng_inbuilt, subkey = random.split(rng_inbuilt)
pred_inbuilt = inbuilt_model.apply(
    {"params": params_inbuilt}, 
    X_test_seq, 
    rngs={"lstm": subkey}
)
pred_inbuilt = jnp.squeeze(pred_inbuilt)

# FIXED: Match PyTorch print format
print(f"Predictions with Custom Model for new sequence: {pred_custom.tolist()}")
print(f"Predictions with In-Built Model: {pred_inbuilt.tolist()}")

# ---------------------------
# Plot the Predictions
# ---------------------------
# FIXED: Match PyTorch plot style (no figsize, simpler labels)
plt.figure()
plt.plot(pred_custom, label="custom model")
plt.plot(pred_inbuilt, label="inbuilt model")
plt.legend()
plt.show()