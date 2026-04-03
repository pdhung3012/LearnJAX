import jax
import jax.numpy as jnp
from jax import grad, jit
from flax import linen as nn
from flax.training import train_state
import optax
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)

# Convert to numpy, normalize to [-1, 1] (matching transforms.Normalize((0.5,), (0.5,))), reshape to (N, 28, 28, 1)
train_images = np.array(train_dataset.data.numpy(), dtype=np.float32) / 255.0
train_images = (train_images - 0.5) / 0.5  # Normalize to [-1, 1]
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = np.array(test_dataset.data.numpy(), dtype=np.float32) / 255.0
test_images = (test_images - 0.5) / 0.5  # Normalize to [-1, 1]
test_images = test_images.reshape(-1, 28, 28, 1)

def data_loader(images, batch_size, shuffle=True):
    n = len(images)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield jnp.array(images[batch_idx])

# Define an Autoencoder model
class Autoencoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Encoder
        x = nn.Conv(32, (3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(64, (3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # Decoder
        x = nn.ConvTranspose(32, (3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(1, (3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.sigmoid(x)
        return x

# Initialize the model, loss function, and optimizer
model = Autoencoder()
dummy_input = jnp.ones([1, 28, 28, 1])
variables = model.init(jax.random.PRNGKey(0), dummy_input)

optimizer = optax.adam(learning_rate=0.001)
state = train_state.TrainState.create(
    apply_fn=model.apply, params=variables['params'], tx=optimizer
)

@jit
def train_step(state, batch):
    def loss_fn(params):
        reconstructed = model.apply({'params': params}, batch)
        return jnp.mean((reconstructed - batch) ** 2)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jit
def compute_loss(params, batch):
    reconstructed = model.apply({'params': params}, batch)
    return jnp.mean((reconstructed - batch) ** 2)

# Training loop
epochs = 10
for epoch in range(epochs):
    for batch in data_loader(train_images, batch_size=64, shuffle=True):
        state, loss = train_step(state, batch)

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Detect anomalies using reconstruction error
threshold = 0.1
anomalies = []
for batch in data_loader(test_images, batch_size=64, shuffle=False):
    reconstructed = model.apply({'params': state.params}, batch)
    loss = jnp.mean((reconstructed - batch) ** 2)

    if loss.item() > threshold:
        anomalies.append(batch)

# Visualize anomalies
if anomalies:
    anomaly_image = anomalies[0][0].squeeze()
    print(f"Anomaly image shape: {anomaly_image.shape}")
    plt.imshow(np.array(anomaly_image), cmap='gray')
    plt.show()
else:
    print("No anomalies detected.")
