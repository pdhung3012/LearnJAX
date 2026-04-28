"""Freeze fixtures for the BERT case.

Builds a small BertModel with a fixed torch seed, dumps:
- pt_weights/  : the PT model directory (loadable by FlaxBertModel.from_pretrained(..., from_pt=True))
- inputs.npz   : input_ids + attention_mask
- expected.npz : last_hidden_state from a PT forward
"""
from pathlib import Path
import numpy as np
import torch
from transformers import BertConfig, BertModel

HERE = Path(__file__).parent


def make_inputs():
    return {
        "input_ids":      np.array([[1, 5, 9, 13, 17, 21, 25, 29]], dtype=np.int64),
        "attention_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1]],     dtype=np.int64),
    }


def build_pt_model(seed: int = 0) -> BertModel:
    config = BertConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=32,
    )
    torch.manual_seed(seed)
    model = BertModel(config)
    model.eval()
    return model


def main():
    inputs = make_inputs()
    model = build_pt_model()

    # Save PT weights so the JAX side can load via from_pt=True.
    pt_dir = HERE / "pt_weights"
    model.save_pretrained(pt_dir)

    with torch.no_grad():
        out = model(
            input_ids=torch.from_numpy(inputs["input_ids"]),
            attention_mask=torch.from_numpy(inputs["attention_mask"]),
        )
    expected = {"last_hidden_state": out.last_hidden_state.numpy()}

    np.savez(HERE / "inputs.npz", **inputs)
    np.savez(HERE / "expected.npz", **expected)
    print("bert: fixtures written")
    for k, v in inputs.items():
        print(f"  input[{k}]: shape={v.shape}, dtype={v.dtype}")
    for k, v in expected.items():
        print(f"  expected[{k}]: shape={v.shape}, dtype={v.dtype}")
    print(f"  pt_weights/ contents: {sorted(p.name for p in pt_dir.iterdir())}")


if __name__ == "__main__":
    main()
