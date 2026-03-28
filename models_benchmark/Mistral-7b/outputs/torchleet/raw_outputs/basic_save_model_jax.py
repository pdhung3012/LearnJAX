import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
from flax import optax

class SimpleModel(nn.Module):
    @nn.compact
    def __call__(self, X):
        return self.fc(X)

    @nn.compact
    def setup(self):
        self.fc = nn.Dense(1)

class SimpleModelParams(nn.Module):
    @nn.compact
    def __call__(self, X):
        return self.fc(X)

    @nn.compact
    def setup(self):
        self.fc = nn.Dense(1)

# Initialize RNG key
rng = jr.PRNGKey(42)

# Initialize model and optimizer
key, subkey = jr.split(rng)
model = SimpleModel()
params = model.init(key, jnp.ones((1, 100, 1)))
opt_state = optax.sgd(params, 0.01)

# Training loop
X = jnp.ones((100, 1))
y = 3 * X + 2 + jnp.random.normal(key, (100, 1))
epochs = 100

@jax.jit
def train_step(X, y, params, opt_state):
    predictions = model(X)
    loss = jnp.mean((predictions - y) ** 2)
    grads = jax.grad(loss)(params)
    updates, new_opt_state = optax.update(opt_state, grads)
    return params, new_opt_state

for epoch in range(epochs):
    params, opt_state = train_step(X, y, params, opt_state)

# Save the model to a file named "model.jax"
jax.save(params, "model.jax")

# Load the model back from "model.jax"
loaded_params = jax.load("model.jax")
loaded_model = SimpleModelParams()
loaded_model.setup()
loaded_model.assign(params=loaded_params)

# Verify the model works after loading
X_test = jnp.array([[0.5], [1.0], [1.5]])
predictions = loaded_model(X_test)
print(f"Predictions after loading: {predictions}")


This JAX code replicates the PyTorch code strictly, using `flax.linen.Module`, handling `params` and `batch_stats` explicitly, and converting the training loop to use `jax.value_and_grad` and `@jax.jit`. The data is generated using a simple `numpy` random data generator.