"""Hand-written PyTorch — EfficientNet-style MBConv block with Squeeze-and-Excitation.

Architecture (one MBConv block + SE):
    expand:    Conv1x1 -> BN -> SiLU
    depthwise: Conv3x3 groups=C_mid -> BN -> SiLU
    SE block:
        squeeze   = global_avg_pool(x)            # (B, C_mid, 1, 1)
        excite_a  = Conv1x1(C_mid -> C_se) -> SiLU
        excite_b  = Conv1x1(C_se -> C_mid) -> Sigmoid
        x         = x * excite_b                  # channelwise re-scaling
    project:   Conv1x1 -> BN
    + residual

New patterns this case adds beyond the other 4:
- **Squeeze-and-Excitation** = global pool -> 2 Conv1x1 layers (with
  reduction ratio) -> sigmoid -> channel-wise multiply. Fully present in
  EfficientNet/MobileNetV3.
- **SiLU** activation (= x * sigmoid(x)). Cheap LLMs sometimes confuse
  with GELU.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EffNetMBBlockConfig:
    in_channels: int = 16
    expansion_factor: int = 4
    out_channels: int = 16
    se_ratio: int = 4               # SE bottleneck reduction
    image_size: int = 16
    bn_eps: float = 1e-5


class SqueezeExcite(nn.Module):
    def __init__(self, in_channels: int, se_channels: int):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, se_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(se_channels, in_channels, kernel_size=1)

    def forward(self, x):
        squeeze = x.mean(dim=[2, 3], keepdim=True)   # (B, C, 1, 1)
        gate = F.silu(self.fc1(squeeze))
        gate = torch.sigmoid(self.fc2(gate))
        return x * gate


class EffNetMBBlock(nn.Module):
    def __init__(self, config: EffNetMBBlockConfig):
        super().__init__()
        self.config = config
        c_in = config.in_channels
        c_mid = c_in * config.expansion_factor
        c_out = config.out_channels
        c_se = max(1, c_mid // config.se_ratio)
        self.expand_conv = nn.Conv2d(c_in, c_mid, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(c_mid, eps=config.bn_eps)
        self.dw_conv = nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1,
                                 groups=c_mid, bias=False)
        self.dw_bn = nn.BatchNorm2d(c_mid, eps=config.bn_eps)
        self.se = SqueezeExcite(c_mid, c_se)
        self.project_conv = nn.Conv2d(c_mid, c_out, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(c_out, eps=config.bn_eps)
        self.use_residual = (c_in == c_out)

    def forward(self, x):
        identity = x
        out = F.silu(self.expand_bn(self.expand_conv(x)))
        out = F.silu(self.dw_bn(self.dw_conv(out)))
        out = self.se(out)
        out = self.project_bn(self.project_conv(out))
        if self.use_residual:
            out = out + identity
        return out


class EffNetMBModel(nn.Module):
    def __init__(self, config: EffNetMBBlockConfig):
        super().__init__()
        self.config = config
        self.block = EffNetMBBlock(config)

    def forward(self, x):
        return self.block(x)


def _seed_bn_buffers(model: nn.Module, seed: int):
    rng = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.data = torch.rand(m.num_features, generator=rng) - 0.5
            m.running_var.data = 0.5 + torch.rand(m.num_features, generator=rng)


def build_pt_model(seed: int = 0) -> EffNetMBModel:
    config = EffNetMBBlockConfig()
    torch.manual_seed(seed)
    model = EffNetMBModel(config)
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
