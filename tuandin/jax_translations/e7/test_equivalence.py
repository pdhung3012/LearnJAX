"""e7 equivalence test: train -> save -> load -> predict.

Same Linear(1,1) forward equivalence as e1; we also verify that the JAX
translation's pickle round-trip preserves params, which is the key claim
of the script.
"""
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close, torch_linear_to_jax


def main():
    rng = np.random.default_rng(0)
    W = rng.uniform(-1, 1, (1, 1)).astype(np.float32)
    b = rng.uniform(-1, 1, (1,)).astype(np.float32)
    X = rng.uniform(0, 1, (3, 1)).astype(np.float32)

    pt = nn.Linear(1, 1)
    with torch.no_grad():
        pt.weight.copy_(torch.from_numpy(W)); pt.bias.copy_(torch.from_numpy(b))
    out_pt = pt(torch.from_numpy(X)).detach().numpy()

    Wj, bj = torch_linear_to_jax(pt.weight, pt.bias)
    params = {"W": Wj, "b": bj}

    # Pickle round-trip (the JAX equivalent of torch.save/load).
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "params.pkl")
        with open(path, "wb") as f:
            pickle.dump(jax.device_get(params), f)
        with open(path, "rb") as f:
            loaded = pickle.load(f)

    out_jax = np.asarray(jnp.asarray(X) @ loaded["W"] + loaded["b"])
    assert_close(out_pt, out_jax, atol=1e-6, name="forward_after_pickle_roundtrip")
    print("[e7] PASS")


if __name__ == "__main__":
    main()
