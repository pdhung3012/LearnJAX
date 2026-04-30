"""Freeze fixtures for simple_bn_block.

Builds the hand-written PT model, populates BN running stats deterministically,
saves state_dict (incl. BN buffers) + config.json, and runs forward.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from pytorch_code import build_pt_model


def make_inputs():
    rng = np.random.default_rng(42)
    pixel_values = rng.standard_normal((1, 3, 16, 16)).astype(np.float32)
    return {"pixel_values": pixel_values}


def _save_pt_weights(model, pt_dir, config_dict):
    pt_dir.mkdir(parents=True, exist_ok=True)
    # Filter out integer counters that safetensors can't save (or that we
    # don't need). num_batches_tracked is an int64 0-d tensor.
    state_dict = {k: v for k, v in model.state_dict().items()
                  if not k.endswith("num_batches_tracked")}
    save_file(state_dict, str(pt_dir / "model.safetensors"))
    with open(pt_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)


def main():
    inputs = make_inputs()
    model = build_pt_model()
    cfg = model.config
    config_dict = {
        "in_channels": cfg.in_channels,
        "hidden_channels": cfg.hidden_channels,
        "num_classes": cfg.num_classes,
        "image_size": cfg.image_size,
        "bn_eps": cfg.bn_eps,
    }
    _save_pt_weights(model, HERE / "pt_weights", config_dict)
    with torch.no_grad():
        out = model(torch.from_numpy(inputs["pixel_values"]))
    expected = {"logits": out.numpy()}
    np.savez(HERE / "inputs.npz", **inputs)
    np.savez(HERE / "expected.npz", **expected)
    print("simple_bn_block: fixtures written")


if __name__ == "__main__":
    main()
