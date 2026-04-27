"""JAX translation of m3: CIFAR-10 CNN trained with several initialization schemes.

Faithful to PyTorch:
- Same architecture: Conv(3->32, 3x3, pad 1) -> ReLU -> Conv(32->64, 3x3, pad 1)
  -> ReLU -> MaxPool(2,2) -> FC(64*16*16->128) -> ReLU -> FC(128->10).
- Optimizer Adam(lr=1e-3), CrossEntropy, batch_size=32, 10 epochs.
- Five inits: Vanilla (Flax default = LeCun normal for kernels, zeros for biases),
  Kaiming (he_normal, fan_out, ReLU), Xavier (xavier_normal), Zeros, Random (normal).
- Reuses torchvision's CIFAR-10 download to keep the data identical to PyTorch.

Speed notes: dominated by data loading on CPU. The pure compute (forward/back/step)
inside jit is comparable or faster than PyTorch on CPU. Wall-clock per epoch on CPU
will be similar because torchvision DataLoader feeds both implementations.
"""
import numpy as np
import jax
import jax.lax as lax
import jax.numpy as jnp
import flax.linen as nn
import optax
import torch


# ---- Contract API used by test_equivalence.py ------------------------------
def compute(inputs):
    """VanillaCNN forward with caller-supplied weights (PyTorch NCHW + (out, in, kH, kW)).

    Inputs:
        c1_w (32, 3, 3, 3), c1_b (32,)
        c2_w (64, 32, 3, 3), c2_b (64,)
        f1_w (128, 64*16*16), f1_b (128,)
        f2_w (10, 128), f2_b (10,)
        x (B, 3, 32, 32) NCHW
    Returns: dict with "logits" (B, 10).
    """
    # NCHW -> NHWC for JAX/Flax conv.
    x = jnp.asarray(np.transpose(inputs["x"], (0, 2, 3, 1)))
    c1_k = jnp.asarray(np.transpose(inputs["c1_w"], (2, 3, 1, 0)))  # HWIO
    c2_k = jnp.asarray(np.transpose(inputs["c2_w"], (2, 3, 1, 0)))
    def conv(x, k, b):
        return lax.conv_general_dilated(
            x, k, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"),
        ) + jnp.asarray(b)
    h = jax.nn.relu(conv(x, c1_k, inputs["c1_b"]))
    h = jax.nn.relu(conv(h, c2_k, inputs["c2_b"]))
    h = nn.max_pool(h, (2, 2), strides=(2, 2))
    # Match PyTorch's view: NHWC -> NCHW -> flatten.
    h = jnp.transpose(h, (0, 3, 1, 2)).reshape(h.shape[0], -1)
    h = jax.nn.relu(h @ jnp.asarray(inputs["f1_w"].T) + jnp.asarray(inputs["f1_b"]))
    return {"logits": np.asarray(h @ jnp.asarray(inputs["f2_w"].T) + jnp.asarray(inputs["f2_b"]))}
import torchvision
import torchvision.transforms as transforms


class VanillaCNN(nn.Module):
    kernel_init: callable = nn.initializers.lecun_normal()
    bias_init: callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        # x: (B, 32, 32, 3) — channels-last for Flax conv.
        x = nn.Conv(32, (3, 3), padding="SAME",
                    kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), padding="SAME",
                    kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(128, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.relu(x)
        x = nn.Dense(10, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return x


def torch_to_numpy_image(batch_images):
    # torchvision DataLoader yields (B, 3, 32, 32). Flax expects (B, 32, 32, 3).
    return np.transpose(batch_images.numpy(), (0, 2, 3, 1))


def train_test_loop(name, init_kernel, init_bias, train_loader, test_loader, epochs=10):
    print(f"_________{name}_______________________")
    model = VanillaCNN(kernel_init=init_kernel, bias_init=init_bias)
    rng = jax.random.PRNGKey(0)
    sample = jnp.zeros((1, 32, 32, 3))
    params = model.init(rng, sample)
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    def loss_fn(params, images, labels):
        logits = model.apply(params, images)
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    @jax.jit
    def train_step(params, opt_state, images, labels):
        loss, grads = jax.value_and_grad(loss_fn)(params, images, labels)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_step(params, images):
        return jnp.argmax(model.apply(params, images), axis=-1)

    last_loss = None
    for epoch in range(epochs):
        for image, label in train_loader:
            x = jnp.asarray(torch_to_numpy_image(image))
            y = jnp.asarray(label.numpy())
            params, opt_state, last_loss = train_step(params, opt_state, x, y)
        print(f"Training loss at epoch {epoch} = {float(last_loss)}")

    correct, total = 0, 0
    for image, label in test_loader:
        x = jnp.asarray(torch_to_numpy_image(image))
        y = label.numpy()
        preds = np.asarray(eval_step(params, x))
        correct += int((preds == y).sum())
        total += int(y.shape[0])
    print(f"Test Accuracy = {(correct * 100) / total}")


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    inits = [
        ("Vanilla", nn.initializers.lecun_normal(), nn.initializers.zeros),
        ("Kaiming", nn.initializers.he_normal(),    nn.initializers.zeros),
        ("Xavier",  nn.initializers.xavier_normal(), nn.initializers.zeros),
        ("Zeros",   nn.initializers.zeros,           nn.initializers.zeros),
        ("Random",  nn.initializers.normal(stddev=1.0), nn.initializers.normal(stddev=1.0)),
    ]
    for name, kinit, binit in inits:
        train_test_loop(name, kinit, binit, train_loader, test_loader)


if __name__ == "__main__":
    main()
