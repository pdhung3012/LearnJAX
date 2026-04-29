"""Hand-written PyTorch — VGG-BN-style small classifier.

Architecture (4 conv blocks + classifier):
    [Conv3x3 -> BN -> ReLU] x 2  (16 channels)  +  MaxPool2x2
    [Conv3x3 -> BN -> ReLU] x 2  (32 channels)  +  MaxPool2x2
    Flatten -> Linear(32*8*8, num_classes)

This is the canonical "CNN with BN" pattern (no residuals, no special
ops). It rounds out the suite with a vanilla architecture so the cheap
LLM is tested on the simplest BN-CNN translation in addition to harder
patterns elsewhere in this tier.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VGGBNSmallConfig:
    in_channels: int = 3
    num_classes: int = 10
    image_size: int = 32
    bn_eps: float = 1e-5


class VGGBNSmall(nn.Module):
    def __init__(self, config: VGGBNSmallConfig):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv2d(config.in_channels, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16, eps=config.bn_eps)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16, eps=config.bn_eps)
        # MaxPool 2x2 stride 2 here brings 32x32 -> 16x16.
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32, eps=config.bn_eps)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(32, eps=config.bn_eps)
        # MaxPool again -> 8x8.
        self.fc = nn.Linear(32 * 8 * 8, config.num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.flatten(1)
        return self.fc(x)


def _seed_bn_buffers(model: nn.Module, seed: int):
    rng = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.data = torch.rand(m.num_features, generator=rng) - 0.5
            m.running_var.data = 0.5 + torch.rand(m.num_features, generator=rng)


def build_pt_model(seed: int = 0) -> VGGBNSmall:
    config = VGGBNSmallConfig()
    torch.manual_seed(seed)
    model = VGGBNSmall(config)
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
