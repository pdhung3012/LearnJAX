import jax
import jax.numpy as jnp
from jax import random, value_and_grad
import pickle

def model(params, x):
    # FIXED: Use matrix multiplication instead of element-wise multiplication
    return jnp.dot(x, params['w']) + params['b']

def mse_loss(params, x, y):
    preds = model(params, x)
    return jnp.mean((preds - y) ** 2)

def train_step(params, x, y, learning_rate=0.01):
    loss, grads = value_and_grad(mse_loss)(params, x, y)
    new_params = {k: params[k] - learning_rate * grads[k] for k in params}
    return new_params, loss

# FIXED: Removed unused generate_random_numbers function

def main():
    key = random.PRNGKey(42)
    
    # FIXED: Use Kaiming uniform initialization instead of normal distribution
    # FIXED: Weight shape changed from (1,) to (1, 1) for proper matrix multiplication
    key, subkey1, subkey2 = random.split(key, 3)
    bound = jnp.sqrt(1.0 / 1)  # Kaiming initialization bound for in_features=1
    params = {
        'w': random.uniform(subkey1, (1, 1), minval=-bound, maxval=bound),
        'b': random.uniform(subkey2, (1,), minval=-bound, maxval=bound)
    }
    
    key, subkey1, subkey2 = random.split(key, 3)
    X = random.uniform(subkey1, (100, 1))
    noise = random.normal(subkey2, (100, 1)) * 0.1
    y = 3 * X + 2 + noise
    
    epochs = 100
    for epoch in range(epochs):
        params, loss = train_step(params, X, y, learning_rate=0.01)
        # FIXED: Removed training progress print to match PyTorch (no output during training)
    
    with open("model.pth", "wb") as f:
        pickle.dump(params, f)
    
    with open("model.pth", "rb") as f:
        loaded_params = pickle.load(f)
    
    X_test = jnp.array([[0.5], [1.0], [1.5]])
    predictions = model(loaded_params, X_test)
    # FIXED: Use f-string format to match PyTorch output
    print(f"Predictions after loading: {predictions}")


if __name__ == "__main__":
    main()