"""Freeze fixtures for the ViT case (symmetric design)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import ViTConfig as HFConfig, ViTModel as HFModel

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from pytorch_code import build_pt_model


def make_inputs():
    rng = np.random.default_rng(42)
    pixel_values = (rng.uniform(0, 1, (1, 3, 32, 32)) * 2 - 1).astype(np.float32)
    return {"pixel_values": pixel_values}


def _save_pt_weights(model, pt_dir, config_dict):
    pt_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(pt_dir / "model.safetensors"))
    with open(pt_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)


def _verify_against_hf(handwritten_model, inputs, expected, atol=5e-6):
    cfg = handwritten_model.config
    hf_cfg = HFConfig(
        hidden_size=cfg.hidden_size, num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        image_size=cfg.image_size, patch_size=cfg.patch_size,
        num_channels=cfg.num_channels, layer_norm_eps=cfg.layer_norm_eps,
    )
    hf_model = HFModel(hf_cfg, add_pooling_layer=False)
    hf_model.load_state_dict(handwritten_model.state_dict(), strict=True)
    hf_model.eval()
    with torch.no_grad():
        hf_out = hf_model(
            pixel_values=torch.from_numpy(inputs["pixel_values"]),
        ).last_hidden_state.numpy()
    diff = np.abs(hf_out - expected["last_hidden_state"]).max()
    assert diff < atol, f"hand-written PT diverges: max abs diff = {diff:.3e}"
    print(f"  HF library equivalence check: ✓ max abs diff = {diff:.3e}")


def main():
    inputs = make_inputs()
    model = build_pt_model()
    cfg = model.config
    config_dict = {
        "hidden_size": cfg.hidden_size, "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "intermediate_size": cfg.intermediate_size,
        "image_size": cfg.image_size, "patch_size": cfg.patch_size,
        "num_channels": cfg.num_channels, "layer_norm_eps": cfg.layer_norm_eps,
    }
    _save_pt_weights(model, HERE / "pt_weights", config_dict)
    with torch.no_grad():
        out = model(pixel_values=torch.from_numpy(inputs["pixel_values"]))
    expected = {"last_hidden_state": out.numpy()}
    np.savez(HERE / "inputs.npz", **inputs)
    np.savez(HERE / "expected.npz", **expected)
    _verify_against_hf(model, inputs, expected)
    print("vit: fixtures written")


if __name__ == "__main__":
    main()
