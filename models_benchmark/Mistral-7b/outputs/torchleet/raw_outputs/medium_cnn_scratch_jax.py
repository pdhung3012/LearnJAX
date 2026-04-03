import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
from flax import optim

# Load CIFAR-10 dataset
batch_size = 64
train_dataset = jax.random.prprng_key(0)
train_data = jax.random.shuffle(jax.random.uniform(train_key, (len(train_dataset), 3, 32, 32)), (jax.random.PRNGKey(0), jnp.arange(len(train_dataset))))
train_dataset = train_data.reshape((len(train_dataset) // batch_size, batch_size, 3, 32, 32))

test_dataset = jax.random.prprng_key(1)
test_data = jax.random.shuffle(jax.random.uniform(test_key, (len(test_dataset), 3, 32, 32)), (jax.random.PRNGKey(1), jnp.arange(len(test_dataset))))
test_dataset = test_data.reshape((len(test_dataset) // batch_size, batch_size, 3, 32, 32))

class Conv2dCustom(nn.Module):
    @nn.compact
    def __call__(self, x):
        batch_size, in_channels, H, W = x.shape
        KH, KW = self.kernel_size
        SH, SW = self.stride
        PH, PW = self.padding

        x_padded = jnp.pad(x, ((0, 0, 0, 0, PW, PW, PH, PH),), mode="constant")

        OH = (H + 2 * PH - KH) // SH + 1
        OW = (W + 2 * PW - KW) // SW + 1

        out = jnp.zeros((batch_size, self.out_channels, OH, OW), dtype=x.dtype)

        for b in jax.pmap(jax.range(batch_size), lambda b: jax.pmap(jax.range(self.out_channels), lambda oc: jax.pmap(jax.range(OH), lambda i: jax.pmap(jax.range(OW), lambda j: jax.pmap(lambda h_start, w_start: jax.ops.index(x_padded[b], jax.ops.index(jnp.broadcast_to(jnp.array([b, oc, i, j, h_start, w_start]), x_padded.shape), 0), 0), axis=-1), axis=-1), axis=-1)):
            region = jnp.squeeze(x_padded[b, :, h_start:h_start + KH, w_start:w_start + KW], axis=-1)
            out[b, oc, i, j] = jnp.sum(region * self.weight[oc]) + self.bias[oc]

        return out

    def setup(self):
        self.kernel_size = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        self.stride = self.stride if self.stride is not None else self.kernel_size
        self.padding = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)

        self.weight = self.param("weight", jnp.random.normal(self.kernel_size[0] * self.out_channels * self.in_channels, (self.out_channels, self.in_channels, *self.kernel_size), dtype=x.dtype))
        self.bias = self.param("bias", jnp.zeros((self.out_channels), dtype=x.dtype))

class MaxPool2dCustom(nn.Module):
    @nn.compact
    def __call__(self, x):
        batch_size, channels, H, W = x.shape
        KH, KW = self.kernel_size
        SH, SW = self.stride

        OH = (H - KH) // SH + 1
        OW = (W - KW) // SW + 1

        out = jnp.zeros((batch_size, channels, OH, OW), dtype=x.dtype)

        for b in jax.pmap(jax.range(batch_size), lambda b: jax.pmap(jax.range(channels), lambda c: jax.pmap(jax.range(OH), lambda i: jax.pmap(jax.range(OW), lambda j: jax.pmap(lambda h_start, w_start: jax.ops.index(x[b, c], jax.ops.index(jnp.broadcast_to(jnp.array([b, c, i, j, h_start, w_start]), x.shape), 0), 0), axis=-1), axis=-1)):
            region = x[b, c, h_start:h_start + KH, w_start:w_start + KW]
            out[b, c, i, j] = jnp.max(region)

        return out

    def setup(self):
        self.kernel_size = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        self.stride = self.stride if self.stride is not None else self.kernel_size

class CNNModel(nn.Module):
    def setup(self):
        self.conv1 = Conv2dCustom(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2dCustom(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = MaxPool2dCustom(kernel_size=2, stride=2)
        self.fc1 = nn.Dense(64 * 16 * 16)
        self.fc2 = nn.Dense(10)

    def __call__(self, x):
        x = jnp.relu(self.conv1(x))
        x = self.pool(x)
        x = x.reshape((x.shape[0], -1))
        x = jnp.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
key = jr.PRNGKey(0)
rng = jr.PRNGSequence(key)
params = CNNModel().init(rng, jnp.ones((1, 3, 32, 32), dtype=jnp.float32))
optimizer = optim.sgd(params, step_size=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_dataset):
        # Forward pass
        outputs = CNNModel()(images)
        loss = jnp.mean(jnp.cross_entropy(labels, jnp.argmax(outputs, axis=-1)))

        # Backward pass and optimization
        grads = jax.value_and_grad(CNNModel().loss)(params, images, labels)
        grads = jax.ops.index_update(grads, jax.ops.index[:, :, 0], jnp.zeros_like(grads[:, :, 0]))
        optimizer, grads = optimizer.update(grads)
        params = jax.tree_multimap(lambda x, g: jax.ops.index_update(x, jax.ops.index[:, :, 0], g), params, grads)

        if i % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Batch [{i}/{len(train_dataset)}], Loss: {loss:.4f}")

# Evaluate on the test set
key = jr.PRNGKey(1)
rng = jr.PRNGSequence(key)
test_params = CNNModel().init(rng, jnp.ones((1, 3, 32, 32), dtype=jnp.float32))
correct = 0
total = 0
for i, (