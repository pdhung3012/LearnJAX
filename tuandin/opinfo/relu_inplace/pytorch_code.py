"""Tests `F.relu_(x)` — in-place ReLU."""
import numpy as np
import torch
import torch.nn.functional as F


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "x": rng.standard_normal((4, 8)).astype(np.float32),
        "scale": rng.uniform(0.5, 2.0, (8,)).astype(np.float32),
    }


def compute(inputs):
    x = torch.from_numpy(inputs["x"]).clone()
    scale = torch.from_numpy(inputs["scale"])
    F.relu_(x)
    out = x * scale
    return {"out": out.numpy()}
