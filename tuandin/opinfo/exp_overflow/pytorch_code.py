"""Tests `torch.exp` of large values — overflows to inf in float32 around x=89."""
import numpy as np
import torch


def make_inputs():
    return {"x": np.array([0.0, 80.0, 100.0, -100.0, 1.0], dtype=np.float32)}


def compute(inputs):
    x = torch.from_numpy(inputs["x"])
    return {"out": torch.exp(x).numpy()}
