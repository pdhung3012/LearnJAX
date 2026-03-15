import jax
import jax.numpy as jnp
from jax import random
import optax  # FIXED: Added missing import
from typing import Any, Tuple

def init_params(key: Any) -> Any:
    keys = random.split(key, 4)
    
    # FIXED: Use Kaiming initialization bounds
    # For layer 1 (2->10): bound = sqrt(1/fan_in) = sqrt(1/2)
    bound1 = jnp.sqrt(1.0 / 2)
    W1 = random.uniform(keys[0], shape=(2, 10), minval=-bound1, maxval=bound1)
    b1 = random.uniform(keys[1], shape=(10,), minval=-bound1, maxval=bound1)
    
    # For layer 2 (10->1): bound = sqrt(1/fan_in) = sqrt(1/10)
    bound2 = jnp.sqrt(1.0 / 10)
    W2 = random.uniform(keys[2], shape=(10, 1), minval=-bound2, maxval=bound2)
    b2 = random.uniform(keys[3], shape=(1,), minval=-bound2, maxval=bound2)
    
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

def predict(params: Any, inputs: jnp.ndarray) -> jnp.ndarray:
    hidden = jnp.dot(inputs, params['W1']) + params['b1']
    hidden = jax.nn.relu(hidden)
    output = jnp.dot(hidden, params['W2']) + params['b2']
    return output

def loss_fn(params: Any, inputs: jnp.ndarray, targets: jnp.ndarray) -> float:
    predictions = predict(params, inputs)
    return jnp.mean((predictions - targets) ** 2)

def main() -> None:
    """Main entry point for the program."""
    key = random.PRNGKey(42)
    key, subkey_params = random.split(key)
    params = init_params(subkey_params)

    key, subkey_X = random.split(key)
    X = random.uniform(subkey_X, shape=(100, 2), minval=0.0, maxval=1.0) * 10
    key, subkey_noise = random.split(key)
    noise = random.normal(subkey_noise, shape=(100, 1))
    y = (X[:, 0:1] + X[:, 1:2] * 2) + noise

    epochs = 1000
    lr = 0.01
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    
    for epoch in range(epochs):
        grads = jax.grad(loss_fn)(params, X, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        if (epoch + 1) % 100 == 0:
            current_loss = loss_fn(params, X, y)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {current_loss:.4f}")
    
    X_test = jnp.array([[4.0, 3.0], [7.0, 8.0]])
    predictions = predict(params, X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

if __name__ == "__main__":
    main()