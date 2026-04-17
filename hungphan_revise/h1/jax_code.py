import jax.numpy as jnp  # MODIFIED: Consistent import of jax.numpy as jnp
from jax import random, value_and_grad
import pickle

def model(params, x):
    return params['w'] * x + params['b']

def mse_loss(params, x, y):
    preds = model(params, x)
    return jnp.mean((preds - y) ** 2)

def train_step(params, x, y, learning_rate=0.01):
    loss, grads = value_and_grad(mse_loss)(params, x, y)
    new_params = {k: params[k] - learning_rate * grads[k] for k in params}
    return new_params, loss

def generate_random_numbers(shape):
    return random.normal(key=random.PRNGKey(0), shape=shape)  # Example method to generate random numbers

def main():
    key = random.PRNGKey(42)
    
    key, subkey1, subkey2 = random.split(key, 3)
    params = {
        'w': random.normal(subkey1, (1,)),
        'b': random.normal(subkey2, (1,))
    }
    
    key, subkey1, subkey2 = random.split(key, 3)
    X = random.uniform(subkey1, (100, 1))
    noise = random.normal(subkey2, (100, 1)) * 0.1
    y = 3 * X + 2 + noise
    
    epochs = 100
    for epoch in range(epochs):
        params, loss = train_step(params, X, y, learning_rate=0.01)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    with open("model.pth", "wb") as f:
        pickle.dump(params, f)
    
    with open("model.pth", "rb") as f:
        loaded_params = pickle.load(f)
    
    X_test = jnp.array([[0.5], [1.0], [1.5]])
    predictions = model(loaded_params, X_test)
    print("Predictions after loading:", predictions)


if __name__ == "__main__":
    main()  # Entry point of the program