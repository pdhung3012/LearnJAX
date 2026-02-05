import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np
import tensorflow_datasets as tfds

# Load CIFAR-10 dataset
def load_data():
    ds_train = tfds.load('cifar10', split='train', batch_size=-1)
    ds_test = tfds.load('cifar10', split='test', batch_size=-1)
    ds_train = tfds.as_numpy(ds_train)
    ds_test = tfds.as_numpy(ds_test)
    
    # Normalize images: ToTensor (0-1) then Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    train_images = ds_train['image'].astype(np.float32) / 255.0
    train_images = (train_images - 0.5) / 0.5
    train_labels = ds_train['label']
    
    test_images = ds_test['image'].astype(np.float32) / 255.0
    test_images = (test_images - 0.5) / 0.5
    test_labels = ds_test['label']
    
    return (jnp.array(train_images), jnp.array(train_labels)), \
           (jnp.array(test_images), jnp.array(test_labels))


# Custom Conv2d layer (JAX/Flax equivalent)
class Conv2dCustom(nn.Module):
    out_channels: int
    kernel_size: tuple
    stride: int = 1
    padding: int = 0
    
    @nn.compact
    def __call__(self, x):
        # x shape: (batch, H, W, C) - NHWC format in JAX
        in_channels = x.shape[-1]
        KH, KW = self.kernel_size
        SH = SW = self.stride
        PH = PW = self.padding
        
        # Initialize weights and biases
        weight = self.param('weight', 
                           lambda rng, shape: jax.random.normal(rng, shape) * 0.1,
                           (KH, KW, in_channels, self.out_channels))
        bias = self.param('bias', nn.initializers.zeros, (self.out_channels,))
        
        # Pad input
        x_padded = jnp.pad(x, ((0, 0), (PH, PH), (PW, PW), (0, 0)), mode='constant')
        
        batch_size, H, W, _ = x.shape
        OH = (H + 2 * PH - KH) // SH + 1
        OW = (W + 2 * PW - KW) // SW + 1
        
        # Use lax.conv_general_dilated for efficiency (mimics the manual convolution)
        out = jax.lax.conv_general_dilated(
            x, weight,
            window_strides=(SH, SW),
            padding=[(PH, PH), (PW, PW)],
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        out = out + bias
        return out


# Custom MaxPool2d layer (JAX/Flax equivalent)
class MaxPool2dCustom(nn.Module):
    kernel_size: tuple
    stride: int = None
    
    @nn.compact
    def __call__(self, x):
        # x shape: (batch, H, W, C) - NHWC format
        KH, KW = self.kernel_size
        stride = self.stride if self.stride is not None else KH
        SH = SW = stride
        
        # Use nn.max_pool
        out = nn.max_pool(x, window_shape=(KH, KW), strides=(SH, SW), padding='VALID')
        return out


# Define the CNN Model (matching PyTorch architecture)
class CNNModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Input: (batch, 32, 32, 3) in NHWC format
        # conv1: 3 -> 32 channels, kernel 3x3, padding 1 -> Output: (batch, 32, 32, 32)
        x = nn.relu(nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x))
        # conv2: 32 -> 64 channels, kernel 3x3, padding 1 -> Output: (batch, 32, 32, 64)
        x = nn.relu(nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x))
        # pool: 2x2 -> Output: (batch, 16, 16, 64)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # Flatten: (batch, 16*16*64) = (batch, 16384)
        x = x.reshape((x.shape[0], -1))
        # fc1: 16384 -> 128
        x = nn.relu(nn.Dense(128)(x))
        # fc2: 128 -> 10
        x = nn.Dense(10)(x)
        return x


# Initialize model
model = CNNModel()
key = jax.random.PRNGKey(0)
key, init_key = jax.random.split(key)

# Dummy input for initialization (NHWC format)
dummy_input = jnp.ones((1, 32, 32, 3))
variables = model.init(init_key, dummy_input)
params = variables['params']

# Setup optimizer
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)


@jax.jit
def train_step(params, opt_state, images, labels):
    def loss_fn(params):
        logits = model.apply({'params': params}, images)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


@jax.jit
def eval_step(params, images, labels):
    logits = model.apply({'params': params}, images)
    predictions = jnp.argmax(logits, axis=-1)
    correct = jnp.sum(predictions == labels)
    return correct


# Load data
print("Loading CIFAR-10 dataset...", flush=True)
(train_images, train_labels), (test_images, test_labels) = load_data()
print(f"Train images shape: {train_images.shape}, Train labels shape: {train_labels.shape}", flush=True)
print(f"Test images shape: {test_images.shape}, Test labels shape: {test_labels.shape}", flush=True)

# Training loop
epochs = 10
batch_size = 64
num_train = len(train_images)

print("\nStarting training...", flush=True)
for epoch in range(epochs):
    # Shuffle training data
    key, shuffle_key = jax.random.split(key)
    perm = jax.random.permutation(shuffle_key, num_train)
    train_images_shuffled = train_images[perm]
    train_labels_shuffled = train_labels[perm]
    
    # Train for one epoch
    for i in range(0, num_train, batch_size):
        batch_images = train_images_shuffled[i:i+batch_size]
        batch_labels = train_labels_shuffled[i:i+batch_size]
        params, opt_state, loss = train_step(params, opt_state, batch_images, batch_labels)
    
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}", flush=True)

# Evaluate on test set
print("\nEvaluating on test set...", flush=True)
correct = 0
total = 0
num_test = len(test_images)

for i in range(0, num_test, batch_size):
    batch_images = test_images[i:i+batch_size]
    batch_labels = test_labels[i:i+batch_size]
    batch_correct = eval_step(params, batch_images, batch_labels)
    correct += int(batch_correct)
    total += len(batch_labels)

print(f"Test Accuracy: {100 * correct / total:.2f}%", flush=True)
