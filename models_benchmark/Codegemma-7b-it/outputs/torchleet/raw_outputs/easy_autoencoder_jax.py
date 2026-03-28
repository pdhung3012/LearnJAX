import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Define the Autoencoder model using flax.linen
class Autoencoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Encoder
        x = nn.Conv2d(features=32, kernel_size=(3, 3), padding='same')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), (2, 2))  # Downsample to 14x14
        x = nn.Conv2d(features=64, kernel_size=(3, 3), padding='same')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), (2, 2))  # Downsample to 7x7
        encoded = x

        # Decoder
        x = nn.ConvTranspose2d(features=32, kernel_size=(3, 3), padding='same', output_padding=1)(encoded)
        x = nn.relu(x)
        x = nn.ConvTranspose2d(features=1, kernel_size=(3, 3), padding='same', output_padding=1)(x)
        x = nn.sigmoid(x)  # To keep pixel values between 0 and 1
        decoded = x

        return decoded

# Initialize the model, loss function, and optimizer
model = Autoencoder()
loss_fn = nn.MSELoss()
optimizer = optax.adam(learning_rate=0.001)

# Training loop
@jax.jit
def train_step(params, batch):
    images, _ = batch
    params, opt_state = jax.value_and_grad(loss_fn)(params, model(images), images)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

@jax.jit
def evaluate(params, batch):
    images, _ = batch
    reconstructed = model(images)
    loss = loss_fn(reconstructed, images)
    return loss

# Generate random data for training and testing
rng = jax.random.PRNGKey(0)
train_images = jax.random.normal(rng, (1000, 1, 28, 28))
test_images = jax.random.normal(rng, (200, 1, 28, 28))

# Training
num_epochs = 10
opt_state = optimizer.init(model.init(jax.random.PRNGKey(0), jnp.ones((1, 28, 28))))
for epoch in range(num_epochs):
    for images in jax.random_split(train_images, [100] * 10):
        params, opt_state = train_step(params, (images, None))

# Anomaly detection
threshold = 0.1  # Define a threshold for anomaly detection
anomalies = []
for images in jax.random_split(test_images, [100] * 2):
    loss = evaluate(params, (images, None))
    if loss > threshold:
        anomalies.append(images)

# Visualize anomalies
if anomalies:
    anomaly_image = anomalies[0][0]
    print(f"Anomaly image shape: {anomaly_image.shape}")
    plt.imshow(anomaly_image.squeeze().numpy(), cmap='gray')
    plt.show()
else:
    print("No anomalies detected.")