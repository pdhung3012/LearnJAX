"""Hand-written PyTorch — small ResNet18-style network.

Architecture (much smaller than torchvision's resnet18):
    stem: Conv7x7 stride 2 -> BN -> ReLU -> MaxPool 3x3 stride 2
    layer1 (no downsample): 2 BasicBlocks, channels stay at 16
    layer2 (downsample 2x): 2 BasicBlocks, channels 16 -> 32
    avgpool + Linear(32, num_classes)

Each BasicBlock:
    Conv -> BN -> ReLU -> Conv -> BN -> (+ residual) -> ReLU
    The residual path uses a 1x1 conv + BN whenever stride != 1 or
    in_channels != out_channels (the canonical ResNet pattern).

Translation gotchas this case adds beyond simple_bn_block:
- BN INSIDE the residual sum's downsample path (HF resnet18.layer2[0]
  has `downsample.0 = Conv1x1`, `downsample.1 = BatchNorm2d`).
- Residual addition happens AFTER both BN layers, before the final ReLU.
- Stride-2 conv on the second layer's first block.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ResNet18SmallConfig:
    in_channels: int = 3
    num_classes: int = 10
    image_size: int = 32
    bn_eps: float = 1e-5


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, eps: float):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=eps)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, eps=eps),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class ResNet18Small(nn.Module):
    def __init__(self, config: ResNet18SmallConfig):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv2d(config.in_channels, 16, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16, eps=config.bn_eps)
        self.layer1_0 = BasicBlock(16, 16, stride=1, eps=config.bn_eps)
        self.layer1_1 = BasicBlock(16, 16, stride=1, eps=config.bn_eps)
        self.layer2_0 = BasicBlock(16, 32, stride=2, eps=config.bn_eps)
        self.layer2_1 = BasicBlock(32, 32, stride=1, eps=config.bn_eps)
        self.fc = nn.Linear(32, config.num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.layer1_0(x); x = self.layer1_1(x)
        x = self.layer2_0(x); x = self.layer2_1(x)
        x = x.mean(dim=[2, 3])
        return self.fc(x)


def _seed_bn_buffers(model: nn.Module, seed: int):
    rng = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.data = torch.rand(m.num_features, generator=rng) - 0.5
            m.running_var.data = 0.5 + torch.rand(m.num_features, generator=rng)


def build_pt_model(seed: int = 0) -> ResNet18Small:
    config = ResNet18SmallConfig()
    torch.manual_seed(seed)
    model = ResNet18Small(config)
    _seed_bn_buffers(model, seed=seed + 1)
    model.eval()
    return model


def main():
    model = build_pt_model()
    rng = torch.Generator().manual_seed(42)
    pixel_values = torch.randn((1, 3, 32, 32), generator=rng)
    with torch.no_grad():
        out = model(pixel_values)
    print("logits shape:", tuple(out.shape))
    print("checksum:", float(out.sum()))


if __name__ == "__main__":
    main()
