import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import flax
import flax.linen as nn

# Generate random data
rng = jr.PRNGKey(42)
X = jnp.linspace(0, 10, 100)
X = jnp.expand_dims(X, axis=-1)
noise = jr.normal(rng, (100, 1))
y = 2 * X + 3 + noise

# Save the generated data to data.csv (JAX does not support this directly)
# You can use numpy or other libraries to save the data if needed

class LinearRegression(nn.Module):
    @nn.compact
    def __call__(self, X):
        w = self.param('w', nn.initializers.normal(stddev=0.01))
        b = self.param('b', nn.initializers.zeros(1))
        return w * X + b

    @nn.compact
    def setup(self):
        self.params = self.param_attrs

# Initialize the model, loss function, and optimizer
model = LinearRegression()
rng = jr.PRNGKey(42)
params = model.init(rng, jnp.ones((1,)))
state = optax.initialize(jax.grad(model.loss)(params), params)

# Training loop
num_epochs = 1000
batch_size = 32

@jax.jit
def train_step(X_batch, y_batch, state, learning_rate):
    grads, new_state = jax.value_and_grad(model.update)(state, X_batch, y_batch)
    params = optax.apply_updates(params, grads)
    state = optax.apply_updates(state, optax.gradient_descent(learning_rate)(new_state))
    return params, state

for epoch in range(num_epochs):
    X_batch, y_batch = jnp.random.split(jnp.concatenate((X, y), axis=-1), num=2, axis=-1)
    params, state = train_step(X_batch, y_batch, state, 0.01)

    if (epoch + 1) % 100 == 0:
        loss = model.loss(params, jnp.concatenate((X, y), axis=-1))
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Display the learned parameters
w, b = model.params['w'], model.params['b']
print(f"Learned weight: {w.item()}, Learned bias: {b.item()}")

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
with jax.grad():
    predictions = model.apply(params, X_test)
print(f"Predictions for {X_test}: {predictions.tolist()}")


This JAX code generates random data, initializes the model, optimizer, and trains the model using the provided PyTorch training loop structure. Note that JAX does not support saving data to CSV files directly, so you may need to use other libraries or methods to save the data if needed.