"""JAX/Flax translation of the RoBERTa case."""
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from transformers import FlaxRobertaModel

HERE = Path(__file__).parent

_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = FlaxRobertaModel.from_pretrained(str(HERE / "pt_weights"), from_pt=True)
    return _MODEL


def compute(inputs):
    """FlaxRobertaModel forward."""
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
