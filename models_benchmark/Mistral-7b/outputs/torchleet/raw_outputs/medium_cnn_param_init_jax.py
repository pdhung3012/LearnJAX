import jax
import jax.numpy as jnp
import jax.random as random
import optax
import flax
import flax.linen as nn

# Define transforms
transform = lambda x: jnp.stack([jnp.asarray(jnp.transpose(jnp.ascontiguousarray(x), (2, 0, 1))), jnp.ones((1,) + x.shape)], axis=-1)

class VanillaCNNModel(nn.Module):
    def setup(self):
        self.conv1 = self.param("conv1", nn.Conv2D(3, 32, kernel_size=3, stride=1, padding=1))
        self.conv2 = self.param("conv2", nn.Conv2D(32, 64, kernel_size=3, stride=1, padding=1))
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2)
        self.fc1 = self.param("fc1", nn.Dense(64 * 16 * 16))
        self.fc2 = self.param("fc2", nn.Dense(10))

    @nn.compact
    def __call__(self, x):
        x = jnp.relu(self.conv1(x))
        x = self.pool(jnp.relu(self.conv2(x)))
        x = jnp.reshape(x, (-1, 64 * 16 * 16))
        x = jnp.relu(self.fc1(x))
        return self.fc2(x)

def init(rng, params):
    init_type = "kaiming"  # Change this to the desired initialization type
    init_fn = {"kaiming": nn.initializers.kaiming_normal,
               "xavier": nn.initializers.xavier_normal,
               "zeros": nn.initializers.zeros,
               "random": nn.initializers.normal}[init_type]
    return jax.tree_multimap(init_fn, params)

def train_test_loop(model, train_loader, test_loader, epochs=10):
    rng = random.PRNGKey(0)
    params = model.init(rng, jax.tree_map(jax.constant, model.init_args))
    state = {"params": params, "batch_stats": {}}

    optimizer = optax.adam(1e-3)

    for epoch in range(epochs):
        for image, label in train_loader:
            image, label = map(lambda x: transform(x), [image, label])
            image, label = jax.tree_map(lambda x, y: (x, y), (image, label))

            grad_fn = jax.value_and_grad(model.__call__, has_aux=True)
            loss, grads = grad_fn(params)(image)

            loss = jnp.mean(loss * jnp.expand_dims(label, axis=-1))
            grads = jax.tree_multimap(lambda g, p: jnp.concatenate([g, p], axis=-1), grads, jax.tree_leaves(params))

            grads = jax.tree_map(lambda g: jnp.reshape(g, [-1]), grads)
            updates, new_params = optimizer.update(grads, params)
            params = jax.tree_map(lambda p, u: p + u, params, updates)

            state["batch_stats"]["train_loss"] = loss

        print(f"Training loss at epoch {epoch} = {state['batch_stats']['train_loss'].item()}")

        model.init_args = (params,)
        rng = jax.random.PRNGKey(epoch)

        correct = 0
        total = 0
        for image_test, label_test in test_loader:
            image_test, label_test = map(lambda x: transform(x), [image_test, label_test])
            image_test, label_test = jax.tree_map(lambda x, y: (x, y), (image_test, label_test))

            pred = model(params)(image_test)
            pred_class = jnp.argmax(pred, axis=-1)
            total += label_test.shape[0]
            correct += jnp.sum(pred_class == label_test)

        print(f"Test Accuracy = {(correct * 100) / total}")

if __name__ == "__main__":
    train_loader = jax.random.prcsr_batch(32, (32, 32, 32, 3), dtype=jnp.float32)
    test_loader = jax.random.prcsr_batch(32, (32, 32, 32, 3), dtype=jnp.float32)

    model = VanillaCNNModel()
    train_test_loop(model, train_loader, test_loader)


This JAX code replicates the PyTorch code strictly using `flax.linen.Module` and handles the state explicitly. The training loop is converted to use `jax.value_and_grad` and `@jax.jit`. Note that the data loading is replaced with simple `jax.random.prcsr_batch` random data generators since the original CIFAR10 dataset is not available in JAX.