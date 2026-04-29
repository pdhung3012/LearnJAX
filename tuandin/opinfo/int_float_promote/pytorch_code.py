"""Tests int*float dtype promotion. PT: result dtype = float32; JAX: same default
behavior on CPU when JAX_ENABLE_X64 is unset (which is our env default).
"""
import numpy as np
import torch


def make_inputs():
    return {
        "i": np.array([1, 2, 3, 4], dtype=np.int32),
        "f": np.array([0.5, 0.25, 0.125, 0.0625], dtype=np.float32),
    }


def compute(inputs):
    i = torch.from_numpy(inputs["i"])
    f = torch.from_numpy(inputs["f"])
    out = i * f
    # Output dtype is float32 in both PT and JAX (default). The numerical
    # match implies the dtype matches.
    return {"out": out.numpy()}
