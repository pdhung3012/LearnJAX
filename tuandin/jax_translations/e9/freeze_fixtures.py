"""Freeze fixtures for e9: sinusoidal positional embedding table."""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from pytorch_code import SinusoidalPositionalEmbedding


def make_inputs():
    return {
        "max_seq_len": np.array(100, dtype=np.int32),
        "d_model":     np.array(64, dtype=np.int32),
        "seq_len":     np.array(50, dtype=np.int32),
    }


def pytorch_reference(inputs):
    max_seq_len = int(inputs["max_seq_len"])
    d_model = int(inputs["d_model"])
    seq_len = int(inputs["seq_len"])
    pe = SinusoidalPositionalEmbedding(max_seq_len=max_seq_len, d_model=d_model)
    dummy = torch.zeros(1, seq_len, d_model)
    return {
        "pe_table": pe.pe.numpy(),                      # (max_seq_len, d_model)
        "forward":  pe(dummy).numpy(),                  # (1, seq_len, d_model)
    }


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("e9: fixtures written")
