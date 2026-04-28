"""JAX translation of m8: MNIST convolutional autoencoder.

Faithful to PyTorch:
- Encoder: Conv(1->32, 3x3, pad 1), ReLU, MaxPool(2,2), Conv(32->64, 3x3, pad 1),
  ReLU, MaxPool(2,2)  -> 7x7 feature map.
- Decoder: ConvTranspose(64->32, k=3, s=2, pad=1, output_padding=1), ReLU,
  ConvTranspose(32->1, k=3, s=2, pad=1, output_padding=1), Sigmoid.
- Loss: MSE(reconstructed, images). Adam(lr=1e-3). 10 epochs, batch 64.

Note on transposed conv: PyTorch's ConvTranspose2d with stride=2, padding=1,
output_padding=1 doubles the spatial size for an input of even spatial size.
Flax's nn.ConvTranspose with strides=(2,2), padding="SAME" produces the same
output size when the input is even. We use that combination.

Speed notes: forward and backward through small convs jit'd is competitive or
slightly faster than PyTorch on CPU; the data loader is the bottleneck.
"""
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
import torch
import torchvision
import torchvision.transforms as transforms


# ---- Contract API used by test_equivalence.py ------------------------------
def compute(inputs):
    """Encoder forward (Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool).

    Inputs (PyTorch NCHW + (out, in, kH, kW) layout):
        c1_w (32, 1, 3, 3), c1_b (32,)
        c2_w (64, 32, 3, 3), c2_b (64,)
        x (B, 1, 28, 28)
    Returns: {"encoded": (B, 64, 7, 7)} (NCHW, matches PyTorch).
    """
    x = jnp.asarray(np.transpose(inputs["x"], (0, 2, 3, 1)))
    c1_k = jnp.asarray(np.transpose(inputs["c1_w"], (2, 3, 1, 0)))
    c2_k = jnp.asarray(np.transpose(inputs["c2_w"], (2, 3, 1, 0)))
    def conv(x, k, b):
        return lax.conv_general_dilated(
            x, k, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"),
        ) + jnp.asarray(b)
    h = nn.max_pool(jax.nn.relu(conv(x, c1_k, inputs["c1_b"])), (2, 2), strides=(2, 2))
    h = nn.max_pool(jax.nn.relu(conv(h, c2_k, inputs["c2_b"])), (2, 2), strides=(2, 2))
    return {"encoded": np.transpose(np.asarray(h), (0, 3, 1, 2))}


class Autoencoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        # x channels-last (B, 28, 28, 1).
        x = nn.Conv(32, (3, 3), padding="SAME")(x); x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = nn.Conv(64, (3, 3), padding="SAME")(x); x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))  # (B, 7, 7, 64)
        x = nn.ConvTranspose(32, (3, 3), strides=(2, 2), padding="SAME")(x)  # (B, 14, 14, 32)
        x = nn.relu(x)
        x = nn.ConvTranspose(1, (3, 3), strides=(2, 2), padding="SAME")(x)   # (B, 28, 28, 1)
        x = jax.nn.sigmoid(x)
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
    _ = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = Autoencoder()
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.zeros((1, 28, 28, 1)))
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    def loss_fn(params, x):
        rec = model.apply(params, x)
        return jnp.mean((rec - x) ** 2)

    @jax.jit
    def train_step(params, opt_state, x):
        loss, grads = jax.value_and_grad(loss_fn)(params, x)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    epochs = 10
    for epoch in range(epochs):
        last_loss = None
        for images, _ in train_loader:
            # torchvision yields (B, 1, 28, 28); transpose to channels-last.
            x = jnp.asarray(images.numpy()).transpose(0, 2, 3, 1)
            params, opt_state, last_loss = train_step(params, opt_state, x)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {float(last_loss):.4f}")


if __name__ == "__main__":
    main()
