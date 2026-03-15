"""
M5 RNN Sine Wave Prediction - JAX Implementation

Fixed to match PyTorch:
- Regression task (not classification)
- Sine wave data (not random)
- MSE loss (not cross-entropy)
- Output 1 value (not 2 logits)
- Hidden size 50 (not 16)
- Bias terms included
- Batch-of-1 training
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np

# Set random seed to match PyTorch
np.random.seed(42)

# FIXED: Match PyTorch data generation
sequence_length = 10
num_samples = 100

# Create sine wave dataset (matching PyTorch exactly)
X = jnp.linspace(0, 4 * jnp.pi, num_samples).reshape(-1, 1)
y = jnp.sin(X)

print(f"X shape: {X.shape}, y shape: {y.shape}")

# Prepare data for RNN (matching PyTorch's create_in_out_sequences)
def create_in_out_sequences(data, seq_length):
    """Create sliding window sequences"""
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i:i + seq_length])
        out_seq.append(data[i + seq_length])
    return jnp.stack(in_seq), jnp.stack(out_seq)

X_seq, y_seq = create_in_out_sequences(y, sequence_length)
print(f"X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")


# FIXED: RNN with bias terms matching PyTorch nn.RNN
class RNNCell(nn.Module):
    """RNN cell with bias terms matching PyTorch"""
    hidden_size: int
    
    @nn.compact
    def __call__(self, carry, x):
        # Input to hidden: W_ih * x + b_ih
        input_proj = nn.Dense(features=self.hidden_size, name='input_dense')(x)
        # Hidden to hidden: W_hh * h (bias handled by Dense above)
        hidden_proj = nn.Dense(features=self.hidden_size, use_bias=False, name='hidden_dense')(carry)
        
        # Combine and apply tanh
        new_carry = jnp.tanh(input_proj + hidden_proj)
        
        return new_carry, new_carry


# FIXED: RNN Model for regression
class RNNModel(nn.Module):
    """RNN Model matching PyTorch: hidden=50, output=1"""
    hidden_size: int = 50  # FIXED: was 16
    output_size: int = 1   # FIXED: was 2
    
    @nn.compact
    def __call__(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state
        init_carry = jnp.zeros((batch_size, self.hidden_size))
        
        # Create RNN with scan
        RNNLayer = nn.scan(
            RNNCell,
            variable_broadcast='params',
            split_rngs={'params': False},
            in_axes=1,
            out_axes=1
        )
        
        # Apply RNN
        final_carry, all_outputs = RNNLayer(hidden_size=self.hidden_size)(init_carry, x)
        
        # Use last output
        last_output = all_outputs[:, -1, :]
        
        # Output layer
        output = nn.Dense(features=self.output_size, name='fc')(last_output)
        
        return output


# FIXED: MSE loss for regression
def compute_loss(params, model, x, y):
    """MSE loss matching PyTorch"""
    predictions = model.apply({'params': params}, x)
    loss = jnp.mean((predictions - y) ** 2)
    return loss


def train_step(params, opt_state, optimizer, model, x_batch, y_batch):
    """Single training step"""
    loss, grads = jax.value_and_grad(compute_loss)(params, model, x_batch, y_batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def main():
    # Initialize model
    model = RNNModel(hidden_size=50, output_size=1)
    
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, sequence_length, 1))
    variables = model.init(rng, dummy_input)
    params = variables['params']
    
    # Optimizer
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)
    
    # FIXED: Training loop - one sample at a time
    epochs = 500
    
    for epoch in range(epochs):
        # Process one sample at a time like PyTorch
        for i in range(len(X_seq)):
            sequences = X_seq[i:i+1]  # (1, 10, 1)
            labels = y_seq[i:i+1]     # (1, 1)
            
            params, opt_state, loss = train_step(
                params, opt_state, optimizer, model, sequences, labels
            )
        
        # Print every epoch
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {float(loss):.4f}")
    
    # Testing
    print("\nTesting on new data...")
    X_test = jnp.linspace(4 * jnp.pi, 5 * jnp.pi, 10).reshape(1, 10, 1)
    
    predictions = model.apply({'params': params}, X_test)
    print(f"Predictions for new sequence: {predictions.tolist()}")


if __name__ == "__main__":
    main()