"""h6 equivalence test: emulated int8 weight quantization formula.

PyTorch's torch.quantization.quantize_dynamic can't be tested on this CPU
(no qnnpack engine). We test that JAX's per-tensor symmetric Q/DQ produces
the expected values for known inputs.
"""
import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close


def quantize_per_tensor_symmetric(x, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    scale = max(np.max(np.abs(x)) / qmax, 1e-8)
    q = np.round(x / scale).clip(-qmax - 1, qmax)
    return q * scale


def main():
    sys.path.insert(0, str(Path(__file__).parent))
    from jax_code import fake_quantize_dense

    rng = np.random.default_rng(0)
    W = rng.standard_normal((4, 5)).astype(np.float32) * 0.5
    W_ref = quantize_per_tensor_symmetric(W, num_bits=8)
    W_jx = np.asarray(fake_quantize_dense({"W": jnp.asarray(W)})["W"])
    assert_close(W_ref, W_jx, atol=1e-6, name="quantize_dequantize_int8")
    # Quantization error stays small relative to the max abs of W.
    err = float(np.max(np.abs(W - W_jx)))
    assert err < 0.01, f"quantization error too large: {err}"
    print(f"[h6] quant max error: {err:.4f}")
    print("[h6] PASS")


if __name__ == "__main__":
    main()
