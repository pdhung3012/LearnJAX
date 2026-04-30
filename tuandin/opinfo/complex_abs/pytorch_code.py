"""Tests `torch.abs` on complex tensors — returns the magnitude (float)."""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    real = rng.standard_normal((4, 5)).astype(np.float32)
    imag = rng.standard_normal((4, 5)).astype(np.float32)
    return {"z": (real + 1j * imag).astype(np.complex64)}


def compute(inputs):
    z = torch.from_numpy(inputs["z"])
    out = torch.abs(z)
    return {"out": out.numpy()}
