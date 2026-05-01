"""Hand-written PyTorch — minimum case isolating BatchNorm running-stats translation.

Architecture:
    Conv2d -> BatchNorm2d -> ReLU
    Conv2d -> BatchNorm2d -> ReLU
    GlobalAvgPool (mean over spatial dims)
    Linear -> logits

The model is set to eval() so BN reads from `running_mean` / `running_var`
buffers (no batch statistics). The buffers are populated with deterministic
values via `_seed_bn_buffers()` so the output is reproducible without a
training pass.

Translation gotchas this case targets:
- PT BatchNorm2d state_dict keys: weight (=scale), bias, running_mean,
  running_var, num_batches_tracked. JAX needs all four (the last is just a
  counter and can be ignored).
- Flax `nn.BatchNorm(use_running_average=True)` is required at inference;
  forgetting the flag computes batch statistics and produces wrong output
  whose error is hard to spot from shape alone.
- Conv2d weight (out, in, kH, kW) becomes (kH, kW, in, out) in JAX HWIO.
- BN momentum/eps defaults: PT uses momentum=0.1, eps=1e-5 (we use 1e-5;
  momentum doesn't matter at eval).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SimpleBNBlockConfig:
    in_channels: int = 3
    hidden_channels: int = 16
    num_classes: int = 10
    image_size: int = 16
    bn_eps: float = 1e-5


class SimpleBNBlock(nn.Module):
    def __init__(self, config: SimpleBNBlockConfig):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv2d(config.in_channels, config.hidden_channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(config.hidden_channels, eps=config.bn_eps)
        self.conv2 = nn.Conv2d(config.hidden_channels, config.hidden_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(config.hidden_channels, eps=config.bn_eps)
        self.fc = nn.Linear(config.hidden_channels, config.num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.mean(dim=[2, 3])           # global average pool
        return self.fc(x)


def _seed_bn_buffers(model: nn.Module, seed: int):
    """Populate BN running_mean / running_var with deterministic non-trivial
    values (we can't rely on a training pass). Without this both buffers
    would be at their init values (mean=0, var=1) and the BN forward would
    be a near-no-op, which doesn't exercise the running-stats code path."""
    rng = torch.Generator().manual_seed(seed)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.running_mean.data = torch.rand(module.num_features, generator=rng) - 0.5
            module.running_var.data = 0.5 + torch.rand(module.num_features, generator=rng)


def build_pt_model(seed: int = 0) -> SimpleBNBlock:
    config = SimpleBNBlockConfig()
    torch.manual_seed(seed)
    model = SimpleBNBlock(config)
    _seed_bn_buffers(model, seed=seed + 1)
    model.eval()
    return model


def main():
    model = build_pt_model()
    rng = torch.Generator().manual_seed(42)
    pixel_values = torch.randn((1, 3, 16, 16), generator=rng)
    with torch.no_grad():
        out = model(pixel_values)
    print("logits shape:", tuple(out.shape))
    print("checksum:", float(out.sum()))


if __name__ == "__main__":
    main()
