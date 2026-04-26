"""JAX translation of m7: MNIST classifier with timing.

Faithful to PyTorch:
- 28*28 -> 128 (ReLU) -> 10. CrossEntropyLoss + SGD(lr=0.01).
- 5 epochs, batch 64. Print per-epoch loss + epoch time. Test accuracy + test time.

Speed notes:
- Pure compute is small; data loading via torchvision DataLoader is identical to
  PyTorch's. Inside the train step JAX with jit is comparable or slightly faster.
- Per-step `loss.item()` would force a sync — we only sync once per epoch (we
  only need the final batch's loss for printing), so JAX stays asynchronous within
  the loop. PyTorch dispatches eagerly per op so it doesn't enjoy the same
  pipelining.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import torch
import torchvision
import torchvision.transforms as transforms


class SimpleNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(nn.Dense(128)(x))
        x = nn.Dense(10)(x)
        return x


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SimpleNN()
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.zeros((1, 1, 28, 28)))
    opt = optax.sgd(0.01)
    opt_state = opt.init(params)

    def loss_fn(params, x, y):
        logits = model.apply(params, x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def predict(params, x):
        return jnp.argmax(model.apply(params, x), axis=-1)

    epochs = 5
    for epoch in range(epochs):
        start = time.time()
        last_loss = None
        for images, labels in train_loader:
            x = jnp.asarray(images.numpy())  # (B, 1, 28, 28)
            y = jnp.asarray(labels.numpy())
            params, opt_state, last_loss = train_step(params, opt_state, x, y)
        elapsed = time.time() - start
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {float(last_loss):.4f}, Time: {elapsed:.4f}s")

    correct, total = 0, 0
    start = time.time()
    for images, labels in test_loader:
        x = jnp.asarray(images.numpy())
        y = labels.numpy()
        preds = np.asarray(predict(params, x))
        correct += int((preds == y).sum())
        total += int(y.shape[0])
    elapsed = time.time() - start
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%, Testing Time: {elapsed:.4f}s")


if __name__ == "__main__":
    main()
