import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

key = jax.random.PRNGKey(0)

key, subkey = jax.random.split(key)
X = jax.random.uniform(subkey, shape=(100, 1)) * 10
key, subkey = jax.random.split(key)
noise = jax.random.normal(subkey, shape=(100, 1))
y = 2 * X + 3 + noise 

def custom_activation(x):
    return jnp.tanh(x) + x

def model(params, X):
    linear_output = jnp.dot(X, params['w']) + params['b']
    return custom_activation(linear_output)

bound = 1.0
key, subkey = jax.random.split(key)
w = jax.random.uniform(subkey, shape=(1, 1), minval=-bound, maxval=bound)
key, subkey = jax.random.split(key)
b = jax.random.uniform(subkey, shape=(1,), minval=-bound, maxval=bound)
params = {'w': w, 'b': b}

def loss_fn(params, X, y):
    preds = model(params, X)
    return jnp.mean((preds - y) ** 2)

lr = 0.01
epochs = 1000

loss_and_grad = jax.value_and_grad(loss_fn)

@jax.jit
def update(params, X, y):
    loss, grads = loss_and_grad(params, X, y)
    new_params = {
        'w': params['w'] - lr * grads['w'],
        'b': params['b'] - lr * grads['b']
    }
    return new_params

def main():
    global params
    for epoch in range(epochs):
        params = update(params, X, y)
        if (epoch + 1) % 100 == 0:
            current_loss = loss_fn(params, X, y)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {current_loss:.4f}")

    learned_w = params['w'][0, 0]
    learned_b = params['b'][0]
    print(f"Learned weight: {learned_w:.4f}, Learned bias: {learned_b:.4f}")

    plt.figure(figsize=(4, 4))
    X_np = np.array(X)
    y_np = np.array(y)
    plt.scatter(X_np, y_np, label='Training Data')
    
    X_line = np.linspace(0, 10, 100).reshape(-1, 1)
    plt.plot(X_line, learned_w * X_line + learned_b, 'r', label='Model Fit')
    plt.legend()
    plt.show()

    X_test = jnp.array([[4.0], [7.0]])
    predictions = model(params, X_test)
    print(f"Predictions for {np.array(X_test).tolist()}: {np.array(predictions).tolist()}")

if __name__ == "__main__":
    main()