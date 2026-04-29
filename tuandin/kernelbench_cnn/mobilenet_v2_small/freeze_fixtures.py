"""Freeze fixtures for mobilenet_v2_small."""
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
    return {"pixel_values": rng.standard_normal((1, 16, 16, 16)).astype(np.float32)}


def main():
    inputs = make_inputs()
    model = build_pt_model()
    cfg = model.config
    config_dict = {
        "in_channels": cfg.in_channels, "expansion_factor": cfg.expansion_factor,
        "out_channels": cfg.out_channels, "image_size": cfg.image_size,
        "bn_eps": cfg.bn_eps,
    }
    pt_dir = HERE / "pt_weights"
    pt_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {k: v for k, v in model.state_dict().items()
                  if not k.endswith("num_batches_tracked")}
    save_file(state_dict, str(pt_dir / "model.safetensors"))
    with open(pt_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    with torch.no_grad():
        out = model(torch.from_numpy(inputs["pixel_values"]))
    np.savez(HERE / "inputs.npz", **inputs)
    np.savez(HERE / "expected.npz", output=out.numpy())
    print("mobilenet_v2_small: fixtures written")


if __name__ == "__main__":
    main()
