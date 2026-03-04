## STRONG LLM
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Define the CNN model with configurable initializers.
class CNNModel(nn.Module):
    conv_kernel_init: callable = nn.initializers.lecun_normal()
    conv_bias_init: callable = nn.initializers.zeros
    dense_kernel_init: callable = nn.initializers.lecun_normal()
    dense_bias_init: callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        # First convolution layer: 3 channels -> 32 filters.
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                    kernel_init=self.conv_kernel_init, bias_init=self.conv_bias_init)(x)
        x = nn.relu(x)
        # Second convolution layer: 32 -> 64 filters.
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                    kernel_init=self.conv_kernel_init, bias_init=self.conv_bias_init)(x)
        x = nn.relu(x)
        # Max pooling: 2x2 window, stride 2.
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        # Flatten.
        x = x.reshape((x.shape[0], -1))
        # Fully connected layer with 128 units.
        x = nn.Dense(features=128, kernel_init=self.dense_kernel_init, bias_init=self.dense_bias_init)(x)
        x = nn.relu(x)
        # Final fully connected layer with 10 outputs.
        x = nn.Dense(features=10, kernel_init=self.dense_kernel_init, bias_init=self.dense_bias_init)(x)
        return x

# Create a training state with the model and an Adam optimizer.
def create_train_state(rng, model, learning_rate=0.001, batch_size=32):
    dummy_input = jnp.ones([batch_size, 32, 32, 3], jnp.float32)
    params = model.init(rng, dummy_input)
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Define a jitted training step.
@jax.jit
def train_step(state, images, labels):
    def loss_fn(params):
        logits = state.apply_fn(params, images)
        onehot = jax.nn.one_hot(labels, 10)
        loss = optax.softmax_cross_entropy(logits, onehot).mean()
        return loss
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Evaluate the model on the test dataset.
def evaluate_model(state, test_ds):
    correct = 0
    total = 0
    for images, labels in tfds.as_numpy(test_ds):
        logits = state.apply_fn(state.params, images)
        predictions = jnp.argmax(logits, axis=-1)
        correct += np.sum(np.array(predictions) == np.array(labels))
        total += len(labels)
    accuracy = 100 * correct / total
    print(f"Test Accuracy = {accuracy:.2f}%")
    return accuracy

# Preprocessing: convert images to float and normalize to [-1, 1]
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Scale to [0, 1]
    image = (image - 0.5) / 0.5                 # Normalize to [-1, 1]
    return image, label

def main():
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    rng = jax.random.PRNGKey(0)

    # Load CIFAR-10 training dataset in supervised mode.
    train_ds = tfds.load('cifar10', split='train', as_supervised=True, download=True)
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Load CIFAR-10 test dataset in supervised mode.
    test_ds = tfds.load('cifar10', split='test', as_supervised=True, download=True)
    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Define the initializer configurations.
    initializer_configs = {
        "Vanilla": {  # Use Flax defaults.
            "conv_kernel_init": nn.initializers.lecun_normal(),
            "dense_kernel_init": nn.initializers.lecun_normal(),
        },
        "Kaiming": {
            "conv_kernel_init": nn.initializers.kaiming_normal(),
            "dense_kernel_init": nn.initializers.kaiming_normal(),
        },
        "Xavier": {
            "conv_kernel_init": nn.initializers.xavier_normal(),
            "dense_kernel_init": nn.initializers.xavier_normal(),
        },
        "Zeros": {
            "conv_kernel_init": nn.initializers.zeros,
            "dense_kernel_init": nn.initializers.zeros,
        },
        "Random": {
            "conv_kernel_init": nn.initializers.normal(stddev=1.0),
            "dense_kernel_init": nn.initializers.normal(stddev=1.0),
        }
    }

    # Loop over the initialization schemes.
    for name, init_conf in initializer_configs.items():
        print(f"_________{name}_______________________")
        model = CNNModel(conv_kernel_init=init_conf["conv_kernel_init"],
                         dense_kernel_init=init_conf["dense_kernel_init"])
        state = create_train_state(rng, model, learning_rate, batch_size)

        # Training loop.
        for epoch in range(num_epochs):
            for images, labels in tfds.as_numpy(train_ds):
                state, loss = train_step(state, images, labels)
            print(f"Training loss at epoch {epoch} = {loss:.4f}")

        # Evaluate the model on the test set.
        evaluate_model(state, test_ds)

if __name__ == "__main__":
    main()
