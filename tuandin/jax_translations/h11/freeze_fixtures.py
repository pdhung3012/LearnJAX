"""Generate canonical inputs.npz and PyTorch-reference expected.npz for h11
(grouped-query attention).

Note: configuration scalars (d_model, num_query_heads, num_query_groups) are
saved as 0-d numpy arrays so the entire spec lives in inputs.npz.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F


def make_inputs():
    rng = np.random.default_rng(0)
    B, S, d_model = 2, 4, 64
    num_query_heads, num_query_groups = 8, 2
    d_head = d_model // num_query_heads

    return {
        "q":    rng.standard_normal((B, S, d_model)).astype(np.float32),
        "k":    rng.standard_normal((B, S, d_model)).astype(np.float32),
        "v":    rng.standard_normal((B, S, d_model)).astype(np.float32),
        # PyTorch nn.Linear weight shape: (out, in).
        "Q_w":   (rng.standard_normal((num_query_heads * d_head, d_model)).astype(np.float32) * 0.1),
        "K_w":   (rng.standard_normal((num_query_groups * d_head, d_model)).astype(np.float32) * 0.1),
        "V_w":   (rng.standard_normal((num_query_groups * d_head, d_model)).astype(np.float32) * 0.1),
        "W_out": (rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.1),
        # Scalar configuration.
        "d_model":           np.array(d_model, dtype=np.int32),
        "num_query_heads":   np.array(num_query_heads, dtype=np.int32),
        "num_query_groups":  np.array(num_query_groups, dtype=np.int32),
    }


def pytorch_reference(inputs):
    q = torch.from_numpy(inputs["q"])
    k = torch.from_numpy(inputs["k"])
    v = torch.from_numpy(inputs["v"])
    Q_w = torch.from_numpy(inputs["Q_w"])
    K_w = torch.from_numpy(inputs["K_w"])
    V_w = torch.from_numpy(inputs["V_w"])
    W_out = torch.from_numpy(inputs["W_out"])
    d_model = int(inputs["d_model"])
    num_query_heads = int(inputs["num_query_heads"])
    num_query_groups = int(inputs["num_query_groups"])
    d_head = d_model // num_query_heads
    B, S, _ = q.shape

    Q = (q @ Q_w.T).view(B, S, num_query_heads, d_head).transpose(1, 2)
    K = (k @ K_w.T).view(B, S, num_query_groups, d_head).transpose(1, 2)
    V = (v @ V_w.T).view(B, S, num_query_groups, d_head).transpose(1, 2)
    n_rep = num_query_heads // num_query_groups
    K = K.repeat_interleave(n_rep, dim=1)
    V = V.repeat_interleave(n_rep, dim=1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_head)
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, d_model)
    out = out @ W_out.T
    return {"output": out.numpy()}


def main():
    inputs = make_inputs()
    expected = pytorch_reference(inputs)
    np.savez("inputs.npz",   **inputs)
    np.savez("expected.npz", **expected)
    print("h11: wrote inputs.npz and expected.npz")
    for k, v in inputs.items():
        print(f"  input[{k}]: shape={v.shape}, dtype={v.dtype}")
    for k, v in expected.items():
        print(f"  expected[{k}]: shape={v.shape}, dtype={v.dtype}")


if __name__ == "__main__":
    main()
