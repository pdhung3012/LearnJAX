"""Freeze fixtures for h6: per-tensor symmetric int8 quantize-dequantize.

PyTorch's torch.quantization.quantize_dynamic isn't available on this CPU,
so the reference is a numpy-only implementation of the same Q/DQ formula.
"""
import numpy as np


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "W":         (rng.standard_normal((4, 5)) * 0.5).astype(np.float32),
        "num_bits":  np.array(8, dtype=np.int32),
    }


def reference(inputs):
    """Numpy reference for symmetric per-tensor int8 Q/DQ."""
    x = inputs["W"]
    num_bits = int(inputs["num_bits"])
    qmax = 2 ** (num_bits - 1) - 1
    scale = max(np.max(np.abs(x)) / qmax, 1e-8)
    q = np.round(x / scale).clip(-qmax - 1, qmax)
    return {"W_quantized": (q * scale).astype(np.float32)}


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **reference(inputs))
    print("h6: fixtures written")
