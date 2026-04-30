"""Tests `torch.scatter_add` — accumulate src values into x at given indices.

Different from `scatter`: when multiple src positions map to the same
output position, scatter_add SUMS (scatter overwrites with last write).
"""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "x":     rng.standard_normal((4, 6)).astype(np.float32),
        # Index includes duplicates so scatter_add semantics matter.
        "index": np.array([
            [1, 1, 3],
            [0, 4, 4],
            [2, 2, 5],
            [3, 0, 0],
        ], dtype=np.int64),
        "src":   rng.standard_normal((4, 3)).astype(np.float32),
    }


def compute(inputs):
    x = torch.from_numpy(inputs["x"]).clone()
    index = torch.from_numpy(inputs["index"])
    src = torch.from_numpy(inputs["src"])
    out = torch.scatter_add(x, dim=1, index=index, src=src)
    return {"out": out.numpy()}
