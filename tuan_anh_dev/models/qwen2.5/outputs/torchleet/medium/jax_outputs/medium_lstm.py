```python
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.experimental import optimizers
import matplotlib.pyplot as plt

# Generate synthetic sequential data
key = jax.random.PRNGKey(42)
sequence_length = 10
num_samples = 100

# Create a sine wave dataset
X = jnp.linspace(0, 4 * 3.14159, num_samples)[None, :, None]
y = jnp.sin(X)

# Prepare data for LSTM
def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i:i + seq_length])
        out_seq.append(data[i + seq_length])
    return jnp.array(in_seq), jnp.array(out_seq)

X_seq, y_seq = create_in_out_sequences(y, sequence_length)

class CustomLSTMModel:
    def __init__(self, input_dim, hidden_units):
        key = jax.random.PRNGKey(0)
        self.Wxi, self.Whi, self.bi = jax.random.normal(key, (input_dim, hidden_units)), jax.random.normal(key, (hidden_units, hidden_units)), jnp.zeros(hidden_units)
        self.Wxf, self.Whf, self.bf = jax.random.normal(key, (input_dim, hidden_units)), jax.random.normal(key, (hidden_units, hidden_units)), jnp.zeros(hidden_units)
        self.Wxo, self.Who, self.bo = jax.random.normal(key, (input_dim, hidden_units)), jax.random.normal(key, (hidden_units, hidden_units)), jnp.zeros(hidden_units)
        self.Wxc, self.Whc, self.bc = jax.random.normal(key, (input_dim, hidden_units)), jax.random.normal(key, (hidden_units, hidden_units)), jnp.zeros(hidden_units)
        self.fc = jnp.zeros((hidden_units, 1))

    def forward(self, inputs, H_C=None):
        batch_size, seq_len, _ = inputs.shape
        if H_C is None:
            H = jnp.zeros((batch_size, self.hidden_units))
            C = jnp.zeros((batch_size, self.hidden_units))
        else:
            H, C = H_C
        
        all_hidden_states = []
        for t in range(seq_len):  
            X_t = inputs[:, t, :]
            I_t = jax.nn.sigmoid(jnp.dot(X_t, self.Wxi) + jnp.dot(H, self.Whi) + self.bi)
            F_t = jax.nn.sigmoid(jnp.dot(X_t, self.Wxf) + jnp.dot(H, self.Whf) + self.bf)
            O_t = jax.nn.sigmoid(jnp.dot(X_t, self.Wxo) + jnp.dot(H, self.Who) + self.bo)
            C_tilde = jnp.tanh(jnp.dot(X_t, self.Wxc) + jnp.dot(H, self.Whc) + self.bc)
            C = F_t * C + I_t * C_tilde
            H = O_t * jnp.tanh(C)
            all_hidden_states.append(H[None, :, :])
            
        outputs = jnp.concatenate(all_hidden_states, axis=1)
        pred = jnp.dot(outputs, self.fc)
        return pred, (H, C)

class LSTMModel:
    def __init__(self):
        key = jax.random.PRNGKey(0)
        self.lstm_params = jax.random.normal(key, (1, 50, 1))
        self.fc_params = jax.random.normal(key, (50, 1))

    def forward(self, x):
        lstm_out, _ = self.lstm(x, self.lstm_params)
        out = jnp.dot(lstm_out[:, -1, :], self.fc_params)  # Use the last output of the LSTM
        return out

# Initialize the model, loss function, and optimizer
model_custom = CustomLSTMModel(1, 50)
model_inbuilt = LSTMModel()
criterion = lambda pred, target: jnp.mean((pred - target) ** 2)

@jit
def update(params, grads, opt_state, step):
    updates, opt_state = optimizers.adam(step)(grads, params, opt_state)
    return opt_state, opt_state['target']

# Training loop for the custom model
opt_init, opt_update, get_params = optimizers.adam(step_size=0.01)
opt_state_custom = opt_init(model_custom.__dict__)
opt_state_inbuilt = opt_init(model_inbuilt.__dict__)

for epoch in range(500):
    # Forward pass
    state = None
    pred, state = model_custom.forward(X_seq, state)
    loss = criterion(pred[:, -1, :], y_seq) # Use the last output of the LSTM
    # Backward pass and optimization
    grads_custom = grad(lambda m, s: criterion(m.forward(X_seq, s)[0][:, -1, :], y_seq))(model_custom, state)
    opt_state_custom, _ = update(model_custom.__dict__, grads_custom, opt_state_custom, epoch)
    
    # Log progress every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/500], Loss: {loss:.4f}")

# Training loop for the inbuilt model
opt_state_inbuilt, _ = update(model_inbuilt.__dict__, grad(criterion)(model_inbuilt.forward(X_seq), y_seq), opt_state_inbuilt, epoch)

for epoch in range
