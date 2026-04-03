import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Define a simple model
class SimpleModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1)(x)
        return x

# Create and train the model
jax.random.PRNGKey(42)
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optax.sgd(0.01)

# Training loop
X = jnp.random.rand(100, 1)
y = 3 * X + 2 + jnp.random.randn(100, 1) * 0.1
epochs = 100

@jax.jit
def train_step(params, batch):
    x, y = batch
    predictions = model.apply(params, x)
    loss = criterion(predictions, y)
    grad_fn = jax.value_and_grad(loss, has_aux=False)
    params, _ = grad_fn(params, x)
    params = optax.apply_updates(params, optimizer.update(grad_fn(params, x)[1], params))
    return params

for epoch in range(epochs):
    params = train_step(params, (X, y))

# Save the model to a file named "model.pth"
params = jax.tree_util.tree_map(jnp.array, model.init(jax.random.PRNGKey(42), jnp.zeros((1, 1))))
jax.save(params, "model.pth")

# Load the model back from "model.pth"
params = jax.tree_util.tree_map(jnp.array, jax.numpy.load("model.pth"))
model = SimpleModel()

# Verify the model works after loading
X_test = jnp.array([[0.5], [1.0], [1.5]])
predictions = model.apply(params, X_test)
print(f"Predictions after loading: {predictions}")