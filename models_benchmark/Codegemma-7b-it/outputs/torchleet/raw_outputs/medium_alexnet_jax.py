import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Define AlexNet
class AlexNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv2d(96, (11, 11), strides=(4, 4), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2))

        x = nn.Conv2d(256, (5, 5), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2))

        x = nn.Conv2d(384, (3, 3), padding='SAME')(x)
        x = nn.relu(x)

        x = nn.Conv2d(384, (3, 3), padding='SAME')(x)
        x = nn.relu(x)

        x = nn.Conv2d(256, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(4096)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.5)(x)

        x = nn.Dense(4096)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.5)(x)

        x = nn.Dense(10)(x)
        return x

# --- Training setup ---
key = jax.random.PRNGKey(0)
model = AlexNet()
params = model.init(key, jnp.ones((1, 3, 224, 224)))

# Dummy data for training
train_images = jnp.random.normal(key, (100, 3, 224, 224))
train_labels = jnp.random.randint(0, 10, (100,))

# Optimizer
opt_init, opt_update, get_params = optax.rmsprop(1e-3)
opt_state = opt_init(params)

@jax.jit
def train_step(params, opt_state, images, labels):
    def loss_fn(params):
        logits = model.apply({'params': params}, images)
        loss = jnp.mean(optax.softmax_cross_entropy_with_logits(logits, labels))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(params)
    updates, opt_state = opt_update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state

# Training loop
for epoch in range(500):
    loss, params, opt_state = train_step(params, opt_state, train_images, train_labels)
    print(f"Epoch [{epoch+1}/500], Loss: {loss:.4f}")

# Evaluation
test_images = jnp.random.normal(key, (20, 3, 224, 224))
test_labels = jnp.random.randint(0, 10, (20,))

logits = model.apply({'params': params}, test_images)
accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == test_labels)
print(f'Test Accuracy: {accuracy:.2f}')