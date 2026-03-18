# run_onnx.py
import os
import tempfile

import numpy as np
import torch
import jax.numpy as jnp
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
from jax.experimental import jax2tf

from pytorch_refactored import SimpleModel, make_model
from jax_code_fixed import model_apply as jax_model_apply

# Shared fixed params (same as test_equivalence.py)
W_np = np.array([[0.25]], dtype=np.float32)
B_np = np.array([0.10], dtype=np.float32)
X_test_np = np.array([[0.5], [1.0], [1.5]], dtype=np.float32)

_DIR = os.path.dirname(os.path.abspath(__file__))


def inject_pytorch_params(model: SimpleModel, w_np: np.ndarray, b_np: np.ndarray) -> None:
    with torch.no_grad():
        model.fc.weight.data.copy_(torch.from_numpy(w_np).to(model.fc.weight.dtype))
        model.fc.bias.data.copy_(torch.from_numpy(b_np).to(model.fc.bias.dtype))


def make_jax_params(w_np: np.ndarray, b_np: np.ndarray) -> dict:
    return {
        "w": jnp.array(w_np, dtype=jnp.float32),
        "b": jnp.array(b_np, dtype=jnp.float32),
    }


def export_pytorch_to_onnx(onnx_path: str) -> None:
    model = make_model()
    inject_pytorch_params(model, W_np, B_np)
    model.eval()
    dummy_input = torch.from_numpy(X_test_np)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )


def export_jax_to_onnx(onnx_path: str) -> None:
    jax_params = make_jax_params(W_np, B_np)

    def jax_forward(x):
        return jax_model_apply(jax_params, x)

    n = X_test_np.shape[0]
    tf_fn = tf.function(jax2tf.convert(jax_forward, native_serialization=False, enable_xla=False))
    spec = (tf.TensorSpec((n, 1), tf.float32),)
    tf2onnx.convert.from_function(
        tf_fn,
        input_signature=spec,
        opset=13,
        output_path=onnx_path,
    )


def test_onnx_equivalence(tol: float = 1e-5) -> None:
    pt_onnx = os.path.join(_DIR, "pt_model.onnx")
    jax_onnx = os.path.join(_DIR, "jax_model.onnx")

    export_pytorch_to_onnx(pt_onnx)
    export_jax_to_onnx(jax_onnx)

    pt_ort = ort.InferenceSession(pt_onnx)
    jax_ort = ort.InferenceSession(jax_onnx)

    pt_input_name = pt_ort.get_inputs()[0].name
    jax_input_name = jax_ort.get_inputs()[0].name
    pt_out = pt_ort.run(None, {pt_input_name: X_test_np})[0]
    jax_out = jax_ort.run(None, {jax_input_name: X_test_np})[0]

    max_diff = float(np.max(np.abs(pt_out - jax_out)))
    if max_diff > tol:
        raise AssertionError(
            f"ONNX equivalence failed: max_abs_diff={max_diff} > tol={tol}"
            f"\nPyTorch ONNX: {pt_out}\nJAX ONNX:     {jax_out}"
        )

    print(f"PASS test_onnx_equivalence (tol={tol})")
    print(f"  PyTorch ONNX: {pt_out.flatten()}")
    print(f"  JAX ONNX:     {jax_out.flatten()}")
    print(f"  max_abs_diff: {max_diff}")


if __name__ == "__main__":
    test_onnx_equivalence()
