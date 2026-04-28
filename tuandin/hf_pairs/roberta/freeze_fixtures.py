"""Freeze fixtures for the RoBERTa case (symmetric design)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import RobertaConfig as HFRobertaConfig, RobertaModel as HFRobertaModel

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from pytorch_code import build_pt_model


def make_inputs():
    return {
        "input_ids":      np.array([[0, 5, 9, 13, 17, 21, 25, 2]], dtype=np.int64),
        "attention_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1]],     dtype=np.int64),
    }


def _save_pt_weights(model, pt_dir, config_dict):
    pt_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(pt_dir / "model.safetensors"))
    with open(pt_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)


def _verify_against_hf(handwritten_model, inputs, expected, atol=1e-6):
    cfg = handwritten_model.config
    hf_cfg = HFRobertaConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        max_position_embeddings=cfg.max_position_embeddings,
        type_vocab_size=cfg.type_vocab_size,
        pad_token_id=cfg.pad_token_id,
        layer_norm_eps=cfg.layer_norm_eps,
    )
    hf_model = HFRobertaModel(hf_cfg, add_pooling_layer=False)
    hf_model.load_state_dict(handwritten_model.state_dict(), strict=True)
    hf_model.eval()
    with torch.no_grad():
        hf_out = hf_model(
            input_ids=torch.from_numpy(inputs["input_ids"]),
            attention_mask=torch.from_numpy(inputs["attention_mask"]),
        ).last_hidden_state.numpy()
    diff = np.abs(hf_out - expected["last_hidden_state"]).max()
    assert diff < atol, f"hand-written PT diverges from HF Roberta: max abs diff = {diff:.3e}"
    print(f"  HF library equivalence check: ✓ max abs diff = {diff:.3e}")


def main():
    inputs = make_inputs()
    model = build_pt_model()

    cfg = model.config
    config_dict = {
        "vocab_size":              cfg.vocab_size,
        "hidden_size":             cfg.hidden_size,
        "num_hidden_layers":       cfg.num_hidden_layers,
        "num_attention_heads":     cfg.num_attention_heads,
        "intermediate_size":       cfg.intermediate_size,
        "max_position_embeddings": cfg.max_position_embeddings,
        "type_vocab_size":         cfg.type_vocab_size,
        "pad_token_id":            cfg.pad_token_id,
        "layer_norm_eps":          cfg.layer_norm_eps,
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
    print("roberta: fixtures written")


if __name__ == "__main__":
    main()
