import jax
import jax.numpy as jnp
import optax
import numpy as np

# Create a PRNG key
key = jax.random.PRNGKey(42)

# Generate synthetic data
key, subkey = jax.random.split(key)
X = jax.random.uniform(subkey, shape=(100, 2)) * 10

key, subkey = jax.random.split(key)
noise = jax.random.normal(subkey, shape=(100, 1))
y = (X[:, 0] + 2 * X[:, 1]).reshape(-1, 1) + noise

def predict(params, x):
    hidden = jnp.dot(x, params['W1']) + params['b1']
    hidden = jax.nn.relu(hidden)
    output = jnp.dot(hidden, params['W2']) + params['b2']
    return output

# FIX: Use Xavier/Kaiming-like initialization to match PyTorch
def init_params(key):
    keys = jax.random.split(key, 4)
    
    # fc1: (2, 10) - Use uniform(-sqrt(1/2), sqrt(1/2))
    limit1 = jnp.sqrt(1.0 / 2)
    W1 = jax.random.uniform(keys[0], shape=(2, 10), minval=-limit1, maxval=limit1)
    b1 = jax.random.uniform(keys[1], shape=(10,), minval=-limit1, maxval=limit1)
    
    # fc2: (10, 1) - Use uniform(-sqrt(1/10), sqrt(1/10))
    limit2 = jnp.sqrt(1.0 / 10)
    W2 = jax.random.uniform(keys[2], shape=(10, 1), minval=-limit2, maxval=limit2)
    b2 = jax.random.uniform(keys[3], shape=(1,), minval=-limit2, maxval=limit2)
    
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

params = init_params(key)

def loss_fn(params, x, y):
    preds = predict(params, x)
    return jnp.mean((preds - y) ** 2)

optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params)

epochs = 1000
loss_and_grad = jax.value_and_grad(loss_fn)

for epoch in range(epochs):
    loss, grads = loss_and_grad(params, X, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

X_test = jnp.array([[4.0, 3.0], [7.0, 8.0]])
predictions = predict(params, X_test)
print(f"Predictions for {np.array(X_test).tolist()}: {np.array(predictions).tolist()}")
