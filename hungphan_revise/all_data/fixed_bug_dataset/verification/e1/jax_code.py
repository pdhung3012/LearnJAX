import jax
import jax.numpy as jnp
from jax import grad, jit, random

# Generate synthetic data
def generate_data(num_samples=100):
    key = random.PRNGKey(0)
    X = jnp.linspace(0, 10, num_samples).reshape(-1, 1)
    noise = random.normal(key, shape=X.shape)
    y = 2 * X + 3 + noise  
    return X, y

# Linear regression model
def model(params, x):
    return jnp.dot(x, params["w"]) + params["b"]

# Loss function
def loss_fn(params, x, y):
    preds = model(params, x)
    return jnp.mean((preds - y) ** 2)

# Gradient computation
@jit
def compute_gradient(params, x, y):
    return grad(loss_fn)(params, x, y)

# Training step
@jit
def train_step(params, x, y):
    grads = compute_gradient(params, x, y)
    return {
        "w": params["w"] - 0.01 * grads["w"],
        "b": params["b"] - 0.01 * grads["b"]
    }

# Training loop
def train_model(X, y, num_epochs=1000):
    bound = 1.0  # For in_features=1, bound = 1.
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    w = random.uniform(subkey, shape=(1, 1), minval=-bound, maxval=bound)
    key, subkey = random.split(key)
    b = random.uniform(subkey, shape=(1,), minval=-bound, maxval=bound)
    params = {"w": w, "b": b}
    
    for epoch in range(num_epochs):
        loss, grads = jax.value_and_grad(loss_fn)(params, X, y)
        params = {
            "w": params["w"] - 0.01 * grads["w"],
            "b": params["b"] - 0.01 * grads["b"]
        }

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")
    return params

# Main function
def main():
    X, y = generate_data(100)
    learned_params = train_model(X, y)
    learned_w = learned_params["w"][0, 0]
    learned_b = learned_params["b"][0]
    print(f"Learned weight: {learned_w:.4f}, Learned bias: {learned_b:.4f}")
    
    X_test = jnp.array([[4.0], [7.0]])
    predictions = model(learned_params, X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

if __name__ == "__main__":
    main()