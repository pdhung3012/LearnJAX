"""JAX/Flax translation of the ViT case.

FlaxViTModel expects pixel_values in NCHW (transformers transposes internally
for the conv stem), matching the PT side.
"""
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from transformers import FlaxViTModel

HERE = Path(__file__).parent

_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = FlaxViTModel.from_pretrained(str(HERE / "pt_weights"), from_pt=True)
    return _MODEL


def compute(inputs):
    """FlaxViTModel forward."""
    model = _load_model()
    out = model(pixel_values=jnp.asarray(inputs["pixel_values"]))
    return {"last_hidden_state": np.asarray(out.last_hidden_state)}


if __name__ == "__main__":
    inputs = {k: v for k, v in np.load(HERE / "inputs.npz").items()}
    out = compute(inputs)
    print("last_hidden_state shape:", out["last_hidden_state"].shape)
    print("checksum:", float(out["last_hidden_state"].sum()))
