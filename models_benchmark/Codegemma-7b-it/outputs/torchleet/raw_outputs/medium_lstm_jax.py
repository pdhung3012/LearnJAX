import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Generate synthetic sequential data
torch.manual_seed(42)
sequence_length = 10
num_samples = 100

# Create a sine wave dataset
X = torch.linspace(0, 4 * 3.14159, steps=num_samples).unsqueeze(1)
y = torch.sin(X)

# Prepare data for LSTM
def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i:i + seq_length])
        out_seq.append(data[i + seq_length])
    return jnp.array(in_seq), jnp.array(out_seq)

X_seq, y_seq = create_in_out_sequences(y, sequence_length)

class CustomLSTMModel(nn.Module):
    @nn.compact
    def __call__(self, inputs, H_C=None):
        batch_size, seq_len, _ = inputs.shape
        if not H_C:
            H = jnp.random.randn(batch_size, self.hidden_units)
            C = jnp.random.randn(batch_size, self.hidden_units)
        else:
            H, C = H_C
            
        all_hidden_states = []
        for t in range(seq_len):  
            X_t = inputs[:, t, :]
            I_t = jax.nn.sigmoid(nn.Dense(self.hidden_units)(X_t) + nn.Dense(self.hidden_units)(H) + self.bi)
            F_t = jax.nn.sigmoid(nn.Dense(self.hidden_units)(X_t) + nn.Dense(self.hidden_units)(H) + self.bf)
            O_t = jax.nn.sigmoid(nn.Dense(self.hidden_units)(X_t) + nn.Dense(self.hidden_units)(H) + self.bo)
            C_tilde = jax.nn.tanh(nn.Dense(self.hidden_units)(X_t) + nn.Dense(self.hidden_units)(H) + self.bc)
            C = F_t * C + I_t * C_tilde
            H = O_t * jax.nn.tanh(C)
            all_hidden_states.append(H.unsqueeze(1))
            
        outputs = jnp.concatenate(all_hidden_states, axis=1)
        pred = nn.Dense(1)(outputs)
        return pred, (H, C)

# Define the LSTM Model
class LSTMModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        out, _ = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)(x)
        out = nn.Dense(1)(out[:, -1, :])  # Use the last output of the LSTM
        return out

# Initialize the model, loss function, and optimizer
model_custom = CustomLSTMModel(input_dim=1, hidden_units=50)
model_inbuilt = LSTMModel()
criterion = nn.MSELoss()
optimizer_custom = optax.adam(learning_rate=0.01)
optimizer_inbuilt = optax.adam(learning_rate=0.01)

# Training loop for the custom model
@jax.jit
def train_step(params, batch):
    X, y = batch
    def loss_fn(params):
        pred, _ = model_custom(X, params)
        loss = criterion(pred[:, -1, :], y)
        return loss, (pred, loss)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    params, (pred, loss) = grad_fn(params, (X, y))
    params = optax.apply_updates(params, optimizer_custom.update(loss, params))
    return params, (pred, loss)

@jax.jit
def test_step(params, batch):
    X, y = batch
    pred, _ = model_custom(X, params)
    loss = criterion(pred[:, -1, :], y)
    return loss, (pred, loss)

epochs = 500
for epoch in range(epochs):
    # Forward pass
    params = model_custom.init(jax.random.PRNGKey(0), X_seq)
    for i in range(X_seq.shape[0]):
        params, (pred, loss) = train_step(params, (X_seq[i], y_seq[i]))
    # Log progress every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Training loop for the inbuilt model
@jax.jit
def train_step_inbuilt(params, batch):
    X, y = batch
    def loss_fn(params):
        pred = model_inbuilt(X, params)
        loss = criterion(pred, y)
        return loss, (pred, loss)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    params, (pred, loss) = grad_fn(params, (X, y))
    params = optax.apply_updates(params, optimizer_inbuilt.update(loss, params))
    return params, (pred, loss)

@jax.jit
def test_step_inbuilt(params, batch):
    X, y = batch
    pred = model_inbuilt(X, params)
    loss = criterion(pred, y)
    return loss, (pred, loss)

epochs = 500
for epoch in range(epochs):
    # Forward pass
    params = model_inbuilt.init(jax.random.PRNGKey(0), X_seq)
    for i in range(X_seq.shape[0]):
        params, (pred, loss) = train_step_inbuilt(params, (X_seq[i], y_seq[i]))
    # Log progress every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Testing on new data
test_steps = 100  # Ensure this is greater than sequence_length
X_test = torch.linspace(0, 5 * 3.14159, steps=test_steps).unsqueeze(1)
y_test = torch.sin(X_test)

# Create test input sequences
X_test_seq, _ = create_in_out_sequences(y_test, sequence_length)

params = model_custom.init(jax.random.PRNGKey(0), X_test_seq)
pred_custom, _ = model_custom(X_test_seq, params)

params = model_inbuilt.init(jax.random.PRNGKey(0), X_test_seq)
pred_inbuilt = model_inbuilt(X_test_seq, params)
pred_custom = jnp.flatten(pred_custom[:, -1, :])
pred_inbuilt = pred_inbuilt.squeeze()
print(f"Predictions with Custom Model for new sequence: {pred_custom.tolist()}")
print(f"Predictions with In-Built Model: {pred_inbuilt.tolist()}")

#Plot the predictions
import matplotlib.pyplot as plt
plt.figure()
# plt.plot(y_test, label="Ground Truth")
plt.plot(pred_custom, label="custom model")
plt.plot(pred_inbuilt, label="inbuilt model")
plt.legend()
plt.show()