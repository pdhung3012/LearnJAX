"""Freeze fixtures for the Wav2Vec2 case (symmetric design)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import Wav2Vec2Config as HFConfig, Wav2Vec2Model as HFModel

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from pytorch_code import build_pt_model


def make_inputs():
    rng = np.random.default_rng(42)
    input_values = (rng.uniform(0, 1, (1, 256)) * 2 - 1).astype(np.float32)
    return {"input_values": input_values}


def _save_pt_weights(model, pt_dir, config_dict):
    pt_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(pt_dir / "model.safetensors"))
    with open(pt_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)


def _verify_against_hf(handwritten_model, inputs, expected, atol=5e-6):
    cfg = handwritten_model.config
    hf_cfg = HFConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        conv_dim=list(cfg.conv_dim), conv_kernel=list(cfg.conv_kernel),
        conv_stride=list(cfg.conv_stride), conv_bias=cfg.conv_bias,
        feat_extract_norm=cfg.feat_extract_norm,
        num_conv_pos_embeddings=cfg.num_conv_pos_embeddings,
        num_conv_pos_embedding_groups=cfg.num_conv_pos_embedding_groups,
        layer_norm_eps=cfg.layer_norm_eps,
        feat_proj_layer_norm=True,
        apply_spec_augment=cfg.apply_spec_augment,
        do_stable_layer_norm=False,
    )
    hf_model = HFModel(hf_cfg)
    hf_model.load_state_dict(handwritten_model.state_dict(), strict=False)
    hf_model.eval()
    with torch.no_grad():
        hf_out = hf_model(
            input_values=torch.from_numpy(inputs["input_values"])
        ).last_hidden_state.numpy()
    diff = np.abs(hf_out - expected["last_hidden_state"]).max()
    assert diff < atol, f"hand-written PT diverges: max abs diff = {diff:.3e}"
    print(f"  HF library equivalence check: ✓ max abs diff = {diff:.3e}")


def main():
    inputs = make_inputs()
    model = build_pt_model()
    cfg = model.config
    config_dict = {
        "vocab_size": cfg.vocab_size, "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "intermediate_size": cfg.intermediate_size,
        "conv_dim": list(cfg.conv_dim), "conv_kernel": list(cfg.conv_kernel),
        "conv_stride": list(cfg.conv_stride), "conv_bias": cfg.conv_bias,
        "feat_extract_norm": cfg.feat_extract_norm,
        "num_conv_pos_embeddings": cfg.num_conv_pos_embeddings,
        "num_conv_pos_embedding_groups": cfg.num_conv_pos_embedding_groups,
        "layer_norm_eps": cfg.layer_norm_eps,
        "apply_spec_augment": cfg.apply_spec_augment,
    }
    _save_pt_weights(model, HERE / "pt_weights", config_dict)
    with torch.no_grad():
        out = model(input_values=torch.from_numpy(inputs["input_values"]))
    expected = {"last_hidden_state": out.numpy()}
    np.savez(HERE / "inputs.npz", **inputs)
    np.savez(HERE / "expected.npz", **expected)
    _verify_against_hf(model, inputs, expected)
    print("wav2vec2: fixtures written")


if __name__ == "__main__":
    main()
