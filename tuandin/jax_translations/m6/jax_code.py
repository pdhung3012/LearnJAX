"""JAX translation of m6: visualize a CIFAR-10 batch with augmentation.

This case is data-loading + visualization only — no training. We keep the
torchvision pipeline (RandomHorizontalFlip + RandomCrop + ToTensor + Normalize)
since it's the canonical augmentation API; the only change is that we display
the *un-normalized* batch as a grid. The augmentation runs in PyTorch land —
that's faithful to the original; JAX has no first-party augmentation library.

Speed notes: identical to PyTorch (the work is in transforms + matplotlib).
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms


def imshow(img):
    img = img / 2 + 0.5  # un-normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    _ = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    images, _ = next(iter(train_loader))
    imshow(torchvision.utils.make_grid(images))


if __name__ == "__main__":
    main()
