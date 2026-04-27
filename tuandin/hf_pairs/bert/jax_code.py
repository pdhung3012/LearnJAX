"""JAX/Flax translation of the BERT case.

Loads the PyTorch weights from `pt_weights/` via FlaxBertModel.from_pretrained
with `from_pt=True` (HuggingFace's built-in PT→Flax weight converter), then
runs the Flax forward.

Speed notes: Flax BERT-base-equivalent forward is well-optimised on CPU; for
this tiny config (2 layers, hidden=64) the difference is dominated by jit
compile cost. JAX should be on par with PyTorch eager once warm.
"""
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from transformers import FlaxBertModel

HERE = Path(__file__).parent

# Cache the loaded model so multiple compute() calls don't re-load weights.
_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = FlaxBertModel.from_pretrained(str(HERE / "pt_weights"), from_pt=True)
    return _MODEL


# ---- Contract API used by test_equivalence.py ------------------------------
def compute(inputs):
    """Run FlaxBertModel forward.

    Args:
        inputs: dict with "input_ids" (B, S) int64, "attention_mask" (B, S) int64.
    Returns:
        dict with "last_hidden_state" (B, S, hidden_size).
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
