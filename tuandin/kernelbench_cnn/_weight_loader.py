"""Shared helper: load saved PT model directory's safetensors weights into
a flat dict[str, np.ndarray].

For BatchNorm-bearing cases, the dict includes both parameters
(weight, bias) AND buffers (running_mean, running_var) — the JAX side
needs all four to reproduce eval-mode BatchNorm.
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
