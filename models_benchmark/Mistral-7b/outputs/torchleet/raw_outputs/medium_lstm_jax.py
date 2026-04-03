import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
from flax import optax

# Generate synthetic sequential data
rng = jr.PRNGKey(42)
sequence_length = 10
num_samples = 100
X = jnp.linspace(0, 4 * 3.14159, num_samples)
X = jnp.expand_dims(X, axis=1)

# Prepare data for LSTM
def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in jax.range(len(data) - seq_length):
        in_seq.append(data[i:i + seq_length])
        out_seq.append(data[i + seq_length])
    return jnp.stack(in_seq), jnp.stack(out_seq)

X_seq, y_seq = create_in_out_sequences(X, sequence_length)

class CustomLSTMModel(nn.Module):
    @nn.compact
    def __init__(self, input_dim, hidden_units):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.Wxi = self.param("Wxi", jnp.random.normal(size=(input_dim, hidden_units)))
        self.Whi = self.param("Whi", jnp.random.normal(size=(hidden_units, hidden_units)))
        self.bi = self.param("bi", jnp.zeros((hidden_units,)))
        self.Wxf = self.param("Wxf", jnp.random.normal(size=(input_dim, hidden_units)))
        self.Whf = self.param("Whf", jnp.random.normal(size=(hidden_units, hidden_units)))
        self.bf = self.param("bf", jnp.zeros((hidden_units,)))
        self.Wxo = self.param("Wxo", jnp.random.normal(size=(input_dim, hidden_units)))
        self.Who = self.param("Who", jnp.random.normal(size=(hidden_units, hidden_units)))
        self.bo = self.param("bo", jnp.zeros((hidden_units,)))
        self.Wxc = self.param("Wxc", jnp.random.normal(size=(input_dim, hidden_units)))
        self.Whc = self.param("Whc", jnp.random.normal(size=(hidden_units, hidden_units)))
        self.bc = self.param("bc", jnp.zeros((hidden_units,)))
        self.fc = nn.Dense(1)

    @nn.compact
    def __call__(self, inputs, H_C=None):
        batch_size, seq_len = inputs.shape
        if H_C is None:
            H = jnp.zeros((batch_size, self.hidden_units))
            C = jnp.zeros((batch_size, self.hidden_units))
        else:
            H, C = H_C

        all_hidden_states = []
        for t in jax.range(seq_len):
            X_t = inputs[:, t, :]
            I_t = jnp.sigmoid(jnp.matmul(X_t, self.Wxi) + jnp.matmul(H, self.Whi) + self.bi)
            F_t = jnp.sigmoid(jnp.matmul(X_t, self.Wxf) + jnp.matmul(H, self.Whf) + self.bf)
            O_t = jnp.sigmoid(jnp.matmul(X_t, self.Wxo) + jnp.matmul(H, self.Who) + self.bo)
            C_tilde = jnp.tanh(jnp.matmul(X_t, self.Wxc) + jnp.matmul(H, self.Whc) + self.bc)
            C = F_t * C + I_t * C_tilde
            H = O_t * jnp.tanh(C)
            all_hidden_states.append(H)

        outputs = jnp.concatenate(all_hidden_states, axis=1)
        pred = self.fc(outputs)
        return pred, (H, C)

class LSTMModel(nn.Module):
    @nn.compact
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTMCell(input_size=1, hidden_size=50)
        self.fc = nn.Dense(1)

    @nn.compact
    def __call__(self, inputs):
        hidden_state, cell_state = jax.zero_like(inputs)
        for _ in jax.range(inputs.shape[0]):
            hidden_state, cell_state = self.lstm(inputs[_, :], hidden_state, cell_state)
        pred = self.fc(hidden_state)
        return pred

# Initialize the model, loss function, and optimizer
params_custom = CustomLSTMModel(1, 50).init(rng, jnp.ones((1,)))
params_inbuilt = LSTMModel().init(rng, jnp.ones((1,)))
criterion = nn.MSELoss()
optimizer = optax.adam(params_custom)

# Training loop for the custom model
@jax.jit
def train_step(params, X_seq, y_seq):
    state = None
    pred, state = CustomLSTMModel()(X_seq, state)
    loss = criterion(pred[:, -1, :], y_seq)
    grads = jax.grad(CustomLSTMModel().loss)(params)(X_seq, y_seq, state)
    return optimizer.update(params, grads)

for epoch in jax.range(500):
    state = None
    params = train_step(params_custom, X_seq, y_seq)[0]
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{500}], Loss: {criterion(CustomLSTMModel()(X_seq, state)[0], y_seq).item():.4f}")

# Training loop for the inbuilt model
@jax.jit
def train_step_inbuilt(params, X_seq, y_seq):
    pred = LSTMModel()(X_seq)
    loss = criterion(pred, y_seq)
    grads = jax.grad(LSTMModel().loss)(params)(X_seq, y_seq)
    return optimizer.update(params, grads)

for epoch in jax.range(500):
    params = train_step_inbuilt(params_inbuilt, X_seq, y_seq)[0]
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{500}], Loss: {criterion(LSTMModel()(X_seq), y_seq).item():.4f}")

# Testing on new data
test_steps = 100
X_test = jnp.linspace(0, 5 * 3.14159, test_steps)
X_test = jnp.expand_dims(X_test, axis=1)

# Create test input sequences
X_test_seq, _ = create_in_out_sequences(X_test, sequence_length)

pred_custom, _ = CustomLSTMModel()(X_test_seq)
pred_inbuilt = LSTMModel()(X_test_seq)
pred_custom = jnp.squeeze(pred_custom)
pred_inbuilt = pred_inbuilt
print(f"Predictions with Custom Model for new sequence: {pred_custom.tolist()}")
print(f"Predictions with In-Built Model: {pred_inbuilt.tolist()}")

# Plot the predictions
import matplotlib.pyplot as plt
import jax2d

jax2d.plot(X_test, pred_custom, label="custom model")
jax2d.plot(X_test, pred_inbuilt, label="inbuilt model")
jax2d.legend()