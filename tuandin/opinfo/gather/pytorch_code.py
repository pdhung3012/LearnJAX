"""Tests `torch.gather(input, dim, index)` — index gather along an axis.

Common cheap-LLM error: confuse `gather` (index per-output-position) with
`index_select` (single index list applied to all positions). They are
NOT equivalent.
"""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "x":     rng.standard_normal((4, 6)).astype(np.float32),
        "index": rng.integers(0, 6, (4, 3)).astype(np.int64),
    }


def compute(inputs):
    x = torch.from_numpy(inputs["x"])
    index = torch.from_numpy(inputs["index"])
    out = torch.gather(x, dim=1, index=index)   # shape (4, 3)
    return {"out": out.numpy()}
