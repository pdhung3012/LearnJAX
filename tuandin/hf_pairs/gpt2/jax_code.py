"""JAX/Flax translation of the GPT-2 case.

Uses FlaxGPT2Model.from_pretrained(..., from_pt=True) to convert PT weights.
"""
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from transformers import FlaxGPT2Model

HERE = Path(__file__).parent

_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = FlaxGPT2Model.from_pretrained(str(HERE / "pt_weights"), from_pt=True)
    return _MODEL


def compute(inputs):
    """FlaxGPT2Model forward.

    Args:
        inputs: dict with "input_ids" (B, S), "attention_mask" (B, S).
    Returns:
        dict with "last_hidden_state" (B, S, n_embd).
    """
    model = _load_model()
    out = model(
        input_ids=jnp.asarray(inputs["input_ids"]),
        attention_mask=jnp.asarray(inputs["attention_mask"]),
    )
    return {"last_hidden_state": np.asarray(out.last_hidden_state)}


if __name__ == "__main__":
    inputs = {k: v for k, v in np.load(HERE / "inputs.npz").items()}
    out = compute(inputs)
    print("last_hidden_state shape:", out["last_hidden_state"].shape)
    print("checksum:", float(out["last_hidden_state"].sum()))
