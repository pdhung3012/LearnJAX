"""Freeze fixtures for the BERT case (symmetric design).

The reference is now the hand-written PyTorch BertModel in pytorch_code.py.
This file:
  1. Builds the hand-written model with a fixed seed.
  2. Saves its state_dict (as safetensors) and config.json to pt_weights/.
  3. Runs forward -> expected.npz.
  4. (Sanity check) Builds transformers.BertModel with the same config, loads
     the same weights, runs forward, and asserts the outputs match. This
     confirms our hand-written impl is faithful to the HF library version
     so the cheap LLM is translating from a correct source.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import BertConfig as HFBertConfig, BertModel as HFBertModel

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from pytorch_code import build_pt_model


def make_inputs():
    return {
        "input_ids":      np.array([[1, 5, 9, 13, 17, 21, 25, 29]], dtype=np.int64),
        "attention_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1]],     dtype=np.int64),
    }


def _save_pt_weights(model, pt_dir, config_dict):
    pt_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(pt_dir / "model.safetensors"))
    with open(pt_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)


def _verify_against_hf(handwritten_model, inputs, expected, atol=1e-6):
    """Sanity check: load hand-written weights into HF library BertModel,
    confirm outputs match the hand-written model's outputs."""
    cfg = handwritten_model.config
    hf_cfg = HFBertConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        max_position_embeddings=cfg.max_position_embeddings,
        type_vocab_size=cfg.type_vocab_size,
        layer_norm_eps=cfg.layer_norm_eps,
    )
    hf_model = HFBertModel(hf_cfg, add_pooling_layer=False)
    hf_model.load_state_dict(handwritten_model.state_dict(), strict=True)
    hf_model.eval()
    with torch.no_grad():
        hf_out = hf_model(
            input_ids=torch.from_numpy(inputs["input_ids"]),
            attention_mask=torch.from_numpy(inputs["attention_mask"]),
        ).last_hidden_state.numpy()
    diff = np.abs(hf_out - expected["last_hidden_state"]).max()
    assert diff < atol, (
        f"hand-written PT diverges from HF BertModel: max abs diff = {diff:.3e}"
    )
    print(f"  HF library equivalence check: ✓ max abs diff = {diff:.3e}")


def main():
    inputs = make_inputs()
    model = build_pt_model()

    config_dict = {
        "vocab_size":              model.config.vocab_size,
        "hidden_size":             model.config.hidden_size,
        "num_hidden_layers":       model.config.num_hidden_layers,
        "num_attention_heads":     model.config.num_attention_heads,
        "intermediate_size":       model.config.intermediate_size,
        "max_position_embeddings": model.config.max_position_embeddings,
        "type_vocab_size":         model.config.type_vocab_size,
        "layer_norm_eps":          model.config.layer_norm_eps,
    }
    _save_pt_weights(model, HERE / "pt_weights", config_dict)

    with torch.no_grad():
        out = model(
            input_ids=torch.from_numpy(inputs["input_ids"]),
            attention_mask=torch.from_numpy(inputs["attention_mask"]),
        )
    expected = {"last_hidden_state": out.numpy()}
    np.savez(HERE / "inputs.npz", **inputs)
    np.savez(HERE / "expected.npz", **expected)

    _verify_against_hf(model, inputs, expected)
    print("bert: fixtures written")


if __name__ == "__main__":
    main()
