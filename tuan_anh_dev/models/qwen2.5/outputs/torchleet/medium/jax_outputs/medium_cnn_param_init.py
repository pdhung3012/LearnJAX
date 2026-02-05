import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np
import tensorflow_datasets as tfds

# Load CIFAR-10 dataset (equivalent to torchvision.datasets.CIFAR10)
def load_data():
    ds_train = tfds.load('cifar10', split='train', batch_size=-1)
    ds_test = tfds.load('cifar10', split='test', batch_size=-1)
    ds_train = tfds.as_numpy(ds_train)
    ds_test = tfds.as_numpy(ds_test)

    # Normalize: same as PyTorch transforms.ToTensor() + Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    train_images = ds_train['image'].astype(np.float32) / 255.0
    train_images = (train_images - 0.5) / 0.5
    train_labels = ds_train['label']

    test_images = ds_test['image'].astype(np.float32) / 255.0
    test_images = (test_images - 0.5) / 0.5
    test_labels = ds_test['label']

    return (jnp.array(train_images), jnp.array(train_labels)), \
           (jnp.array(test_images), jnp.array(test_labels))

# CNN Model matching PyTorch architecture exactly:
# conv1(3→32, 3x3, pad=1) → relu → conv2(32→64, 3x3, pad=1) → relu → pool(2x2) → flatten → fc1(16384→128) → relu → fc2(128→10)
class VanillaCNNModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(32, (3, 3), strides=(1, 1), padding='SAME')(x))
        x = nn.relu(nn.Conv(64, (3, 3), strides=(1, 1), padding='SAME')(x))
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(128)(x))
        x = nn.Dense(10)(x)
        return x

model = VanillaCNNModel()

# Initialize default parameters
key = jax.random.PRNGKey(0)
key, init_key = jax.random.split(key)
dummy_input = jnp.ones((1, 32, 32, 3))
variables = model.init(init_key, dummy_input)
default_params = variables['params']

# Parameter initialization functions
# Equivalent to PyTorch's config_init() + model.apply(init_fn)
def reinit_params(params, init_type, key):
    """Reinitialize parameters with the specified strategy."""
    new_params = {}
    for layer_name, layer_params in params.items():
        new_layer = {}
        for param_name, param in layer_params.items():
            key, subkey = jax.random.split(key)
            if init_type == 'kaiming':
                if param_name == 'kernel':
                    new_layer[param_name] = nn.initializers.kaiming_normal()(subkey, param.shape)
                else:  # bias
                    new_layer[param_name] = jnp.zeros_like(param)
            elif init_type == 'xavier':
                if param_name == 'kernel':
                    new_layer[param_name] = nn.initializers.xavier_normal()(subkey, param.shape)
                else:  # bias
                    new_layer[param_name] = jnp.zeros_like(param)
            elif init_type == 'zeros':
                new_layer[param_name] = jnp.zeros_like(param)
            elif init_type == 'random':
                new_layer[param_name] = jax.random.normal(subkey, param.shape)
        new_params[layer_name] = new_layer
    return new_params

# Create params for each initialization strategy
key, k1, k2, k3, k4 = jax.random.split(key, 5)
init_configs = {
    'Vanilla': dict(default_params),
    'Kaiming': reinit_params(dict(default_params), 'kaiming', k1),
    'Xavier': reinit_params(dict(default_params), 'xavier', k2),
    'Zeros': reinit_params(dict(default_params), 'zeros', k3),
    'Random': reinit_params(dict(default_params), 'random', k4),
}

# Training and evaluation function
# Equivalent to PyTorch train_test_loop(model, train_loader, test_loader, epochs=10)
def train_test_loop(params, train_images, train_labels, test_images, test_labels, epochs=10):
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)
    batch_size = 32
    num_train = len(train_images)

    @jax.jit
    def train_step(params, opt_state, images, labels):
        def loss_fn(params):
            logits = model.apply({'params': params}, images)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for epoch in range(epochs):
        for i in range(0, num_train, batch_size):
            batch_images = train_images[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            params, opt_state, loss = train_step(params, opt_state, batch_images, batch_labels)
        print(f"Training loss at epoch {epoch} = {loss.item():.4f}")

    # Evaluation
    correct = 0
    total = 0
    num_test = len(test_images)
    for i in range(0, num_test, batch_size):
        batch_images = test_images[i:i+batch_size]
        batch_labels = test_labels[i:i+batch_size]
        logits = model.apply({'params': params}, batch_images)
        predictions = jnp.argmax(logits, axis=-1)
        total += len(batch_labels)
        correct += int(jnp.sum(predictions == batch_labels))
    print(f"Test Accuracy = {(correct * 100) / total:.2f}%")

# Load data
(train_images, train_labels), (test_images, test_labels) = load_data()

# Train with each initialization strategy
for name, params in init_configs.items():
    print(f"_________{name}_______________________")
    train_test_loop(params, train_images, train_labels, test_images, test_labels)
