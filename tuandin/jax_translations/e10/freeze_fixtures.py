"""Generate canonical inputs.npz and PyTorch-reference expected.npz for e10
(scaled-dot-product attention).
"""
import math
import numpy as np
import torch
import torch.nn.functional as F


def make_inputs():
    rng = np.random.default_rng(0)
    B, S, D = 2, 5, 16
    return {
        "q": rng.standard_normal((B, S, D)).astype(np.float32),
        "k": rng.standard_normal((B, S, D)).astype(np.float32),
        "v": rng.standard_normal((B, S, D)).astype(np.float32),
        "mask": np.tril(np.ones((S, S), dtype=np.float32)),  # causal
    }


def pytorch_reference(inputs):
    q = torch.from_numpy(inputs["q"])
    k = torch.from_numpy(inputs["k"])
    v = torch.from_numpy(inputs["v"])
    mask = torch.from_numpy(inputs["mask"])

    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return {
        "output": out.numpy(),
        "attention_weights": attn.numpy(),
    }


def main():
    inputs = make_inputs()
    expected = pytorch_reference(inputs)
    np.savez("inputs.npz",   **inputs)
    np.savez("expected.npz", **expected)
    print("e10: wrote inputs.npz and expected.npz")
    for k, v in inputs.items():
        print(f"  input[{k}]: shape={v.shape}, dtype={v.dtype}")
    for k, v in expected.items():
        print(f"  expected[{k}]: shape={v.shape}, dtype={v.dtype}")


if __name__ == "__main__":
    main()
