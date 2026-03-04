import jax
import jax.numpy as jnp
from jax import grad, jit, random, vmap
import optax


def model_fn(params, x):
    w, b = params
    return jnp.dot(x, w) + b

# Define a simple model
class LinearModel:
    def __init__(self, key):
        key, subkey = random.split(key)
        w = random.uniform(subkey, (1, 1), minval=-1.0, maxval=1.0)
        key, subkey = random.split(key)
        b = random.uniform(subkey, (1,), minval=-1.0, maxval=1.0)
        self.params = {"w": w, "b": b}

    def __call__(self, x):
        return jnp.dot(x, self.params["w"]) + self.params["b"]

# Loss function
def huber_loss(params, x, y, delta=1.0):
    preds = jnp.dot(x, params["w"]) + params["b"]
    error = jnp.abs(preds - y)
    loss = jnp.where(error <= delta,
                     0.5 * error**2, 
                     delta * (error - 0.5 * delta))
    return jnp.mean(loss)

# Update function using functional programming
def update(params, x, y, learning_rate=0.01):
    loss_value, grads = jax.value_and_grad(huber_loss)(params, x, y, 1.0)
    params["w"] = params["w"] - learning_rate * grads["w"]
    params["b"] = params["b"] - learning_rate * grads["b"]
    return params

# Training function
def train_model(model, x, y, epochs=1000):
    for epoch in range(epochs):
        model.params = update(model.params, x, y, learning_rate=0.01)
        if (epoch + 1) % 100 == 0:
            current_loss = huber_loss(model.params, x, y, 1.0)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {current_loss:.4f}")
    return model

def main():
    # Generate synthetic data
    key = random.PRNGKey(0)  # MODIFIED: Explicit PRNG key
    model = LinearModel(key)
    
    # Generate synthetic data
    key, subkey = random.split(key)
    x = random.uniform(subkey, shape=(100, 1)) * 10
    key, subkey = random.split(key)
    noise = random.normal(subkey, shape=(100, 1))
    y = 2 * x + 3 + noise

    # Train the model
    model = train_model(model, x, y, epochs=1000)

    x = jnp.array([[4.0], [7.0]])
    # Test the model
    predictions = model(x)
    print(f"Predictions for {x.tolist()}: {predictions.tolist()}")
    print(f"Trained weights: {model.params['w']}, bias: {model.params['b']}")

if __name__ == "__main__":
    main()