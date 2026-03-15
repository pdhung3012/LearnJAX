import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import pickle

# FIXED: Implement proper Language Model architecture
class LanguageModel(nn.Module):
    vocab_size: int
    embed_size: int
    hidden_size: int
    num_layers: int
    
    def setup(self):
        # Define layers in setup to avoid NameInUseError
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_size)
        self.lstm_cells = [nn.LSTMCell(features=self.hidden_size) for _ in range(self.num_layers)]
        self.fc_out = nn.Dense(self.vocab_size)
    
    def __call__(self, x):
        # Embedding layer: (batch, seq_length) -> (batch, seq_length, embed_size)
        embedded = self.embedding(x)
        
        batch_size, seq_len, _ = embedded.shape
        
        # Initialize LSTM states for all layers
        hidden_states = [jnp.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
        cell_states = [jnp.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
        
        # Process sequence through LSTM layers
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            for layer in range(self.num_layers):
                (cell_states[layer], hidden_states[layer]), x_t = self.lstm_cells[layer](
                    (cell_states[layer], hidden_states[layer]), 
                    x_t
                )
        
        # Use last hidden state for prediction
        last_hidden = hidden_states[-1]
        
        # Output layer: (batch, hidden_size) -> (batch, vocab_size)
        output = self.fc_out(last_hidden)
        
        # Softmax activation
        return jax.nn.softmax(output, axis=-1)


def loss_fn(params, model, X, y):
    """Compute cross-entropy loss for language modeling."""
    # Get predictions: (batch, vocab_size)
    preds = model.apply({'params': params}, X)
    # Compute cross-entropy loss
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(preds, y))


# ============================================================================
# CUSTOM DYNAMIC QUANTIZATION (mimics PyTorch's quantize_dynamic)
# ============================================================================

def quantize_weight_int8(weight):
    """
    Quantize a weight matrix to int8.
    
    Similar to PyTorch's dynamic quantization:
    - Compute scale and zero_point from weight statistics
    - Quantize to int8 range [-128, 127]
    - Store scale for dequantization during inference
    """
    # Compute scale: map float range to int8 range [-128, 127]
    w_min = jnp.min(weight)
    w_max = jnp.max(weight)
    
    # Scale factor to map [w_min, w_max] -> [-128, 127]
    scale = (w_max - w_min) / 255.0
    scale = jnp.maximum(scale, 1e-8)  # Avoid division by zero
    
    # Zero point (offset)
    zero_point = -128 - w_min / scale
    zero_point = jnp.clip(jnp.round(zero_point), -128, 127)
    
    # Quantize: float -> int8
    quantized = jnp.clip(jnp.round(weight / scale + zero_point), -128, 127).astype(jnp.int8)
    
    return {
        'quantized': quantized,
        'scale': scale,
        'zero_point': zero_point
    }


def dequantize_weight_int8(quantized_data):
    """
    Dequantize int8 weights back to float32 for computation.
    
    During inference: int8 -> float32 using stored scale and zero_point
    """
    quantized = quantized_data['quantized'].astype(jnp.float32)
    scale = quantized_data['scale']
    zero_point = quantized_data['zero_point']
    
    # Dequantize: int8 -> float
    return (quantized - zero_point) * scale


def quantize_params_dynamic(params):
    """
    Apply dynamic quantization to model parameters.
    
    Quantizes Linear layer weights and LSTM weights to int8.
    Similar to PyTorch's quantize_dynamic(model, {nn.Linear, nn.LSTM}, dtype=torch.qint8)
    """
    quantized_params = {}
    
    for key, value in params.items():
        if isinstance(value, dict):
            # Recursively quantize nested dicts
            quantized_params[key] = quantize_params_dynamic(value)
        else:
            # Check if this is a weight matrix (2D array)
            if isinstance(value, jnp.ndarray) and value.ndim == 2:
                # Quantize weight matrices (kernel in Flax Dense/LSTM layers)
                if 'kernel' in str(key) or value.shape[0] > 1:  # Heuristic for weight matrices
                    quantized_params[key] = quantize_weight_int8(value)
                else:
                    # Don't quantize biases or 1D parameters
                    quantized_params[key] = value
            else:
                # Keep other parameters as-is (biases, embeddings, etc.)
                quantized_params[key] = value
    
    return quantized_params


def dequantize_params_for_inference(quantized_params):
    """
    Dequantize parameters for inference.
    
    In practice, you'd keep weights in int8 and dequantize on-the-fly during
    matrix multiplication for speed. Here we dequantize fully for simplicity.
    """
    dequantized_params = {}
    
    for key, value in quantized_params.items():
        if isinstance(value, dict):
            if 'quantized' in value:
                # This is a quantized weight - dequantize it
                dequantized_params[key] = dequantize_weight_int8(value)
            else:
                # Recursively dequantize nested dicts
                dequantized_params[key] = dequantize_params_for_inference(value)
        else:
            # Keep non-quantized parameters as-is
            dequantized_params[key] = value
    
    return dequantized_params


# ============================================================================
# MAIN TRAINING AND QUANTIZATION
# ============================================================================

def main():
    # FIXED: Changed random seed from 0 to 42
    key = jax.random.PRNGKey(42)
    
    # Model hyperparameters (match PyTorch)
    vocab_size = 50
    seq_length = 10
    batch_size = 32
    embed_size = 64
    hidden_size = 128
    num_layers = 2
    num_epochs = 5
    
    # FIXED: Generate integer token IDs instead of continuous floats
    X_train = jax.random.randint(key, (batch_size, seq_length), 0, vocab_size)
    key, subkey = jax.random.split(key)
    # FIXED: Target shape (batch_size,) instead of (batch_size, seq_length)
    y_train = jax.random.randint(subkey, (batch_size,), 0, vocab_size)
    
    # Initialize model
    model = LanguageModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    
    # Initialize parameters
    key, subkey = jax.random.split(key)
    params = model.init(subkey, X_train)['params']
    
    # Initialize optimizer
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)
    
    # Training loop
    for epoch in range(num_epochs):
        # FIXED: Compute loss with proper model parameter passing
        current_loss = loss_fn(params, model, X_train, y_train)
        
        # Compute gradients
        grad = jax.grad(lambda p: loss_fn(p, model, X_train, y_train))(params)
        
        # Update parameters
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        
        # Log progress every epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {current_loss:.4f}")
    
    # ========================================================================
    # QUANTIZATION: Apply dynamic quantization to the TRAINED model
    # ========================================================================
    quantized_params = quantize_params_dynamic(params)
    
    # Save the quantized model
    with open("quantized_language_model.pkl", "wb") as f:
        pickle.dump(quantized_params, f)
    
    # ========================================================================
    # MATCH PYTORCH BUG: Load quantized model and test it
    # ========================================================================
    # Create a NEW model instance (like PyTorch does - this creates RANDOM weights!)
    quantized_model = LanguageModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    
    # Initialize with RANDOM weights
    key, subkey = jax.random.split(key)
    fresh_params = quantized_model.init(subkey, X_train)['params']
    
    # Apply dynamic quantization on the NEW model (quantizes RANDOM weights)
    quantized_model_params = quantize_params_dynamic(fresh_params)
    
    # MATCHING PYTORCH: The load_state_dict line is COMMENTED OUT
    # So we DON'T load the trained weights - we use the random quantized ones!
    # with open("quantized_language_model.pkl", "rb") as f:
    #     quantized_model_params = pickle.load(f)
    
    # Dequantize for inference (using RANDOM weights, not trained!)
    inference_params = dequantize_params_for_inference(quantized_model_params)
    
    # Test the model with new input (using RANDOM weights - matches PyTorch bug!)
    key, subkey = jax.random.split(key)
    test_input = jax.random.randint(subkey, (1, seq_length), 0, vocab_size)
    
    # Run inference with quantized model (RANDOM weights)
    prediction = quantized_model.apply({'params': inference_params}, test_input)
    predicted_token = jnp.argmax(prediction, axis=1).item()
    
    print(f"Prediction for input {test_input.tolist()}: {predicted_token}")


if __name__ == "__main__":
    main()