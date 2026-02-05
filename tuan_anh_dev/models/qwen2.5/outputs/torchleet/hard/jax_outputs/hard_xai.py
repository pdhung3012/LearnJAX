# Explain a CNN model's predictions using Grad-CAM in JAX/Flax
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


IMAGENET_MEAN = jnp.array([0.485, 0.456, 0.406], dtype=jnp.float32)
IMAGENET_STD = jnp.array([0.229, 0.224, 0.225], dtype=jnp.float32)


class BasicBlock(nn.Module):
    features: int
    stride: int = 1
    downsample: bool = False

    def setup(self):
        self.conv1 = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            name="conv1",
        )
        self.bn1 = nn.BatchNorm(name="bn1")
        self.conv2 = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            name="conv2",
        )
        self.bn2 = nn.BatchNorm(name="bn2")
        if self.downsample:
            self.downsample_conv = nn.Conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="VALID",
                use_bias=False,
                name="downsample_conv",
            )
            self.downsample_bn = nn.BatchNorm(name="downsample_bn")

    def __call__(self, x, train=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, use_running_average=not train)
        out = nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, use_running_average=not train)

        if self.downsample:
            identity = self.downsample_conv(identity)
            identity = self.downsample_bn(identity, use_running_average=not train)

        out = out + identity
        return nn.relu(out)


class BasicBlockWithTarget(nn.Module):
    features: int
    stride: int = 1
    downsample: bool = False

    def setup(self):
        self.conv1 = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            name="conv1",
        )
        self.bn1 = nn.BatchNorm(name="bn1")
        self.conv2 = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            name="conv2",
        )
        self.bn2 = nn.BatchNorm(name="bn2")
        if self.downsample:
            self.downsample_conv = nn.Conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="VALID",
                use_bias=False,
                name="downsample_conv",
            )
            self.downsample_bn = nn.BatchNorm(name="downsample_bn")

    def forward_to_conv2(self, x, train=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, use_running_average=not train)
        out = nn.relu(out)

        target_activations = self.conv2(out)

        if self.downsample:
            identity = self.downsample_conv(identity)
            identity = self.downsample_bn(identity, use_running_average=not train)

        return target_activations, identity

    def forward_from_conv2(self, target_activations, identity, train=False):
        out = self.bn2(target_activations, use_running_average=not train)
        out = out + identity
        return nn.relu(out)

    def __call__(self, x, train=False):
        target_activations, identity = self.forward_to_conv2(x, train=train)
        return self.forward_from_conv2(target_activations, identity, train=train)


class ResNet18GradCAM(nn.Module):
    num_classes: int = 1000

    def setup(self):
        self.conv1 = nn.Conv(
            features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding=((3, 3), (3, 3)),
            use_bias=False,
            name="conv1",
        )
        self.bn1 = nn.BatchNorm(name="bn1")

        self.layer1_block1 = BasicBlock(features=64, stride=1, downsample=False, name="layer1_block1")
        self.layer1_block2 = BasicBlock(features=64, stride=1, downsample=False, name="layer1_block2")

        self.layer2_block1 = BasicBlock(features=128, stride=2, downsample=True, name="layer2_block1")
        self.layer2_block2 = BasicBlock(features=128, stride=1, downsample=False, name="layer2_block2")

        self.layer3_block1 = BasicBlock(features=256, stride=2, downsample=True, name="layer3_block1")
        self.layer3_block2 = BasicBlock(features=256, stride=1, downsample=False, name="layer3_block2")

        self.layer4_block1 = BasicBlock(features=512, stride=2, downsample=True, name="layer4_block1")
        self.layer4_block2 = BasicBlockWithTarget(
            features=512,
            stride=1,
            downsample=False,
            name="layer4_block2",
        )

        self.fc = nn.Dense(self.num_classes, name="fc")

    def stem(self, x, train=False):
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        return x

    def forward_to_target(self, x, train=False):
        x = self.stem(x, train=train)

        x = self.layer1_block1(x, train=train)
        x = self.layer1_block2(x, train=train)

        x = self.layer2_block1(x, train=train)
        x = self.layer2_block2(x, train=train)

        x = self.layer3_block1(x, train=train)
        x = self.layer3_block2(x, train=train)

        x = self.layer4_block1(x, train=train)
        target_activations, identity = self.layer4_block2.forward_to_conv2(x, train=train)
        return target_activations, identity

    def forward_from_target(self, target_activations, identity, train=False):
        x = self.layer4_block2.forward_from_conv2(target_activations, identity, train=train)
        x = jnp.mean(x, axis=(1, 2))
        return self.fc(x)

    def __call__(self, x, train=False):
        target_activations, identity = self.forward_to_target(x, train=train)
        return self.forward_from_target(target_activations, identity, train=train)


def _imagenet_normalize(x):
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def _to_pil_image(tensor_hwc):
    arr = np.asarray(tensor_hwc, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def _preprocess(image):
    image = image.resize((224, 224), resample=Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = jnp.asarray(arr, dtype=jnp.float32)
    arr = _imagenet_normalize(arr)
    return arr[None, ...]


def _to_hwio(weight_oi_hw):
    return np.transpose(weight_oi_hw, (2, 3, 1, 0))


def _assign_conv2d(block_params, torch_state, flax_name, torch_name):
    kernel = _to_hwio(torch_state[f"{torch_name}.weight"].detach().cpu().numpy())
    block_params[flax_name]["kernel"] = jnp.asarray(kernel)


def _assign_bn(block_params, block_stats, torch_state, flax_name, torch_name):
    block_params[flax_name]["scale"] = jnp.asarray(torch_state[f"{torch_name}.weight"].detach().cpu().numpy())
    block_params[flax_name]["bias"] = jnp.asarray(torch_state[f"{torch_name}.bias"].detach().cpu().numpy())
    block_stats[flax_name]["mean"] = jnp.asarray(torch_state[f"{torch_name}.running_mean"].detach().cpu().numpy())
    block_stats[flax_name]["var"] = jnp.asarray(torch_state[f"{torch_name}.running_var"].detach().cpu().numpy())


def load_pretrained_torchvision_resnet18(params, batch_stats):
    try:
        from torchvision.models import ResNet18_Weights, resnet18

        torch_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        import torchvision

        torch_model = torchvision.models.resnet18(pretrained=True)

    torch_model.eval()
    torch_state = torch_model.state_dict()

    params_mut = unfreeze(params)
    stats_mut = unfreeze(batch_stats)

    _assign_conv2d(params_mut, torch_state, "conv1", "conv1")
    _assign_bn(params_mut, stats_mut, torch_state, "bn1", "bn1")

    for layer_idx in range(1, 5):
        for block_idx in range(1, 3):
            flax_block = f"layer{layer_idx}_block{block_idx}"
            torch_block = f"layer{layer_idx}.{block_idx - 1}"

            _assign_conv2d(params_mut[flax_block], torch_state, "conv1", f"{torch_block}.conv1")
            _assign_bn(params_mut[flax_block], stats_mut[flax_block], torch_state, "bn1", f"{torch_block}.bn1")
            _assign_conv2d(params_mut[flax_block], torch_state, "conv2", f"{torch_block}.conv2")
            _assign_bn(params_mut[flax_block], stats_mut[flax_block], torch_state, "bn2", f"{torch_block}.bn2")

            if layer_idx > 1 and block_idx == 1:
                _assign_conv2d(
                    params_mut[flax_block],
                    torch_state,
                    "downsample_conv",
                    f"{torch_block}.downsample.0",
                )
                _assign_bn(
                    params_mut[flax_block],
                    stats_mut[flax_block],
                    torch_state,
                    "downsample_bn",
                    f"{torch_block}.downsample.1",
                )

    params_mut["fc"]["kernel"] = jnp.asarray(torch_state["fc.weight"].detach().cpu().numpy().T)
    params_mut["fc"]["bias"] = jnp.asarray(torch_state["fc.bias"].detach().cpu().numpy())

    return freeze(params_mut), freeze(stats_mut)


def compute_grad_cam(model, params, batch_stats, input_tensor, target_class):
    variables = {"params": params, "batch_stats": batch_stats}

    activations, identity = model.apply(
        variables,
        input_tensor,
        train=False,
        method=ResNet18GradCAM.forward_to_target,
    )

    def class_score_fn(target_acts):
        logits = model.apply(
            variables,
            target_acts,
            identity,
            train=False,
            method=ResNet18GradCAM.forward_from_target,
        )
        return logits[0, target_class]

    gradients = jax.grad(class_score_fn)(activations)
    return activations, gradients


# Initialize model
key = jax.random.PRNGKey(0)
model = ResNet18GradCAM(num_classes=1000)
dummy_input = jnp.ones((1, 224, 224, 3), dtype=jnp.float32)
variables = model.init(key, dummy_input, train=False)
params = variables["params"]
batch_stats = variables["batch_stats"]

try:
    params, batch_stats = load_pretrained_torchvision_resnet18(params, batch_stats)
except Exception as exc:
    raise RuntimeError(
        "Could not load pretrained torchvision ResNet18 weights; this translation requires pretrained=True parity."
    ) from exc
print("Loaded torchvision ResNet18 pretrained weights into Flax model.")

# FakeData equivalent: random image, then ImageNet normalize (like dataset transform).
key, img_key = jax.random.split(key)
dataset_tensor = jax.random.uniform(
    img_key,
    (224, 224, 3),
    minval=0.0,
    maxval=1.0,
    dtype=jnp.float32,
)
dataset_tensor = _imagenet_normalize(dataset_tensor)
image = _to_pil_image(dataset_tensor)

# Preprocess image for model (Resize + ToTensor + Normalize).
input_tensor = _preprocess(image)

# Perform a forward pass.
output = model.apply(
    {"params": params, "batch_stats": batch_stats},
    input_tensor,
    train=False,
)
predicted_class = int(jnp.argmax(output, axis=1)[0])

# Grad-CAM for target layer analogous to PyTorch's model.layer4[1].conv2.
activations, gradients = compute_grad_cam(
    model,
    params,
    batch_stats,
    input_tensor,
    predicted_class,
)

# Generate Grad-CAM heatmap.
weights = jnp.mean(gradients, axis=(1, 2), keepdims=True)
heatmap = jnp.sum(weights * activations, axis=-1).squeeze()
heatmap = jnp.maximum(heatmap, 0.0)
heatmap = heatmap / jnp.maximum(jnp.max(heatmap), 1e-8)

# Normalize and overlay heatmap on the original image.
heatmap_resized = jax.image.resize(
    heatmap,
    (image.height, image.width),
    method="bilinear",
)
heatmap_pil = Image.fromarray(
    np.clip(np.asarray(heatmap_resized) * 255.0, 0, 255).astype(np.uint8)
)

# Display result.
plt.imshow(image)
plt.imshow(heatmap_pil, alpha=0.5, cmap="jet")
plt.title(f"Predicted Class: {predicted_class}")
plt.axis("off")
plt.show()
