"""Shared helper: load a HuggingFace-saved PT model directory's safetensors
weights into a flat dict[str, np.ndarray] and load its config.json.

The keys are the raw PyTorch state_dict keys; the model implementation in
jax_code.py is responsible for any layout transposes (Linear (out, in) ->
Flax kernel (in, out), Conv2d (out, in, kH, kW) -> (kH, kW, in, out), etc.).
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open


def load_pt_safetensors(pt_dir) -> dict[str, np.ndarray]:
    pt_dir = Path(pt_dir)
    weights: dict[str, np.ndarray] = {}
    with safe_open(str(pt_dir / "model.safetensors"), framework="numpy") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def load_pt_config(pt_dir) -> dict[str, Any]:
    pt_dir = Path(pt_dir)
    with open(pt_dir / "config.json") as f:
        return json.load(f)
