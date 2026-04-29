"""Hand-written PyTorch — MobileNetV2-style MBConv block.

A single MBConv block with expansion factor t, depthwise 3x3, then project:
    expand:    Conv1x1(C_in -> C_in*t) -> BN -> ReLU6
    depthwise: Conv3x3 groups=C_in*t   -> BN -> ReLU6
    project:   Conv1x1(C_in*t -> C_out) -> BN  (no activation)
    + residual when stride=1 and C_in == C_out.

Patterns this case adds:
- **Depthwise convolution** (`groups=in_channels`). PT and JAX express this
  via `groups` / `feature_group_count`; the cheap LLM has to know to set it.
- **ReLU6** = clamp(x, 0, 6). Both have it (`torch.clamp(x, 0, 6)`,
  `jnp.clip(x, 0, 6)`); cheap LLMs sometimes use plain ReLU and miss
  the 6-cap.
- Final BN-then-residual (no activation after the projection).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MobileNetV2SmallConfig:
    in_channels: int = 16
    expansion_factor: int = 4
    out_channels: int = 16    # equals in_channels so we can use the residual
    image_size: int = 16
    bn_eps: float = 1e-5


def _relu6(x):
    return torch.clamp(x, 0.0, 6.0)


class InvertedResidual(nn.Module):
    """The MBConv block. Stride is fixed to 1 here so we always use the residual."""

    def __init__(self, config: MobileNetV2SmallConfig):
        super().__init__()
        c_in = config.in_channels
        c_mid = c_in * config.expansion_factor
        c_out = config.out_channels
        self.expand_conv = nn.Conv2d(c_in, c_mid, kernel_size=1, bias=False)
        self.expand_bn   = nn.BatchNorm2d(c_mid, eps=config.bn_eps)
        self.dw_conv     = nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1,
                                     groups=c_mid, bias=False)
        self.dw_bn       = nn.BatchNorm2d(c_mid, eps=config.bn_eps)
        self.project_conv = nn.Conv2d(c_mid, c_out, kernel_size=1, bias=False)
        self.project_bn   = nn.BatchNorm2d(c_out, eps=config.bn_eps)
        self.use_residual = (c_in == c_out)

    def forward(self, x):
        identity = x
        out = _relu6(self.expand_bn(self.expand_conv(x)))
        out = _relu6(self.dw_bn(self.dw_conv(out)))
        out = self.project_bn(self.project_conv(out))    # no activation
        if self.use_residual:
            out = out + identity
        return out


class MobileNetV2Small(nn.Module):
    """A single MBConv block exposed as a Module so the case is testable in
    isolation."""

    def __init__(self, config: MobileNetV2SmallConfig):
        super().__init__()
        self.config = config
        self.block = InvertedResidual(config)

    def forward(self, x):
        return self.block(x)


def _seed_bn_buffers(model: nn.Module, seed: int):
    rng = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.data = torch.rand(m.num_features, generator=rng) - 0.5
            m.running_var.data = 0.5 + torch.rand(m.num_features, generator=rng)


def build_pt_model(seed: int = 0) -> MobileNetV2Small:
    config = MobileNetV2SmallConfig()
    torch.manual_seed(seed)
    model = MobileNetV2Small(config)
    _seed_bn_buffers(model, seed=seed + 1)
    model.eval()
    return model


def main():
    model = build_pt_model()
    rng = torch.Generator().manual_seed(42)
    pixel_values = torch.randn((1, 16, 16, 16), generator=rng)
    with torch.no_grad():
        out = model(pixel_values)
    print("output shape:", tuple(out.shape))
    print("checksum:", float(out.sum()))


if __name__ == "__main__":
    main()
