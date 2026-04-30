"""Tests `torch.scatter` (out-of-place form). Inverse of `gather`.

Cheap LLMs often translate scatter incorrectly because JAX's scatter is
expressed via the `.at[]` indexed update API, which has different
semantics for "value per output position" vs "value per index"."""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "x":     np.zeros((4, 6), dtype=np.float32),
        "index": rng.integers(0, 6, (4, 3)).astype(np.int64),
        "src":   rng.standard_normal((4, 3)).astype(np.float32),
    }


def compute(inputs):
    x = torch.from_numpy(inputs["x"]).clone()
    index = torch.from_numpy(inputs["index"])
    src = torch.from_numpy(inputs["src"])
    out = torch.scatter(x, dim=1, index=index, src=src)
    return {"out": out.numpy()}
