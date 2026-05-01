"""Format adapter wrapping torch2jax's output in our compute(inputs) contract.

This is the one-time "format fix" for the deterministic torch2jax tool.
It is NOT counted as a fix step in the pass@1 / fix-step measurement.

Workflow:
1. Import the case's `build_pt_model()` from `pytorch_code.py`.
2. Build the PT model.
3. Load the case's `pt_weights/model.safetensors` into a state_dict of JAX arrays.
4. Use `torch2jax.t2j_module(pt_model)` to produce a JAX-callable.
5. Wrap it in `compute(inputs) -> dict` matching the case's expected output keys.
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch
from safetensors import safe_open
import torch2jax


def _load_pt_state_dict_as_jax(pt_dir: Path) -> dict:
    """Load safetensors into a dict of JAX arrays, keys matching PT state_dict."""
    state_dict = {}
    with safe_open(str(pt_dir / "model.safetensors"), framework="numpy") as f:
        for key in f.keys():
            state_dict[key] = jnp.asarray(f.get_tensor(key))
    return state_dict


def _add_missing_buffers(state_dict: dict, pt_model: torch.nn.Module) -> dict:
    """Some buffers (e.g. BN's num_batches_tracked) are filtered out of our
    safetensors dumps. Refill them from the PT model so the state_dict has
    every key the PT module expects."""
    full = dict(pt_model.state_dict())
    for k, v in full.items():
        if k not in state_dict:
            state_dict[k] = jnp.asarray(v.detach().numpy())
    return state_dict


def make_compute(case_dir: Path, output_keys: tuple[str, ...], input_keys: tuple[str, ...]):
    """Build a `compute(inputs) -> dict` for a given case using torch2jax.

    Args:
        case_dir: directory containing pytorch_code.py and pt_weights/.
        output_keys: keys to put in the returned dict (must match expected.npz).
        input_keys: keys read from inputs.npz (in argument order to forward()).
    """
    sys.path.insert(0, str(case_dir))
    # Each case's pytorch_code module exposes build_pt_model().
    import importlib
    pytorch_code = importlib.import_module("pytorch_code")
    importlib.reload(pytorch_code)
    pt_model = pytorch_code.build_pt_model()

    # Load weights as JAX arrays.
    state_dict = _load_pt_state_dict_as_jax(case_dir / "pt_weights")
    state_dict = _add_missing_buffers(state_dict, pt_model)

    # Build the t2j-translated forward.
    jax_forward = torch2jax.t2j_module(pt_model)

    def compute(inputs: dict) -> dict:
        args = tuple(jnp.asarray(inputs[k]) for k in input_keys)
        out = jax_forward(*args, state_dict=state_dict)
        # `out` may be a single array or a tuple. Normalize to dict.
        if isinstance(out, (tuple, list)):
            assert len(out) == len(output_keys)
            return {k: np.asarray(v) for k, v in zip(output_keys, out)}
        else:
            assert len(output_keys) == 1, f"single output but expected {len(output_keys)} keys"
            return {output_keys[0]: np.asarray(out)}

    return compute
