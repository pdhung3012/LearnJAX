"""Freeze fixtures for m9: multi-head attention forward with shared weights."""
import math
import numpy as np
import torch
import torch.nn.functional as F


def make_inputs():
    rng = np.random.default_rng(0)
    B, S, d_model, num_heads = 2, 4, 8, 2
    return {
        "q":     rng.standard_normal((B, S, d_model)).astype(np.float32),
        "k":     rng.standard_normal((B, S, d_model)).astype(np.float32),
        "v":     rng.standard_normal((B, S, d_model)).astype(np.float32),
        "Q_w":   (rng.standard_normal((d_model, d_model)) * 0.3).astype(np.float32),  # (out, in)
        "K_w":   (rng.standard_normal((d_model, d_model)) * 0.3).astype(np.float32),
        "V_w":   (rng.standard_normal((d_model, d_model)) * 0.3).astype(np.float32),
        "W_out": (rng.standard_normal((d_model, d_model)) * 0.3).astype(np.float32),
        "d_model":   np.array(d_model, dtype=np.int32),
        "num_heads": np.array(num_heads, dtype=np.int32),
    }


def pytorch_reference(inputs):
    q = torch.from_numpy(inputs["q"]); k = torch.from_numpy(inputs["k"]); v = torch.from_numpy(inputs["v"])
    Q_w = torch.from_numpy(inputs["Q_w"]); K_w = torch.from_numpy(inputs["K_w"])
    V_w = torch.from_numpy(inputs["V_w"]); W_out = torch.from_numpy(inputs["W_out"])
    d_model = int(inputs["d_model"]); num_heads = int(inputs["num_heads"])
    d_head = d_model // num_heads
    B, S, _ = q.shape

    Q = (q @ Q_w.T).view(B, S, num_heads, d_head).transpose(1, 2)
    K = (k @ K_w.T).view(B, S, num_heads, d_head).transpose(1, 2)
    V = (v @ V_w.T).view(B, S, num_heads, d_head).transpose(1, 2)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_head)
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, d_model)
    return {"output": (out @ W_out.T).numpy()}


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("m9: fixtures written")
