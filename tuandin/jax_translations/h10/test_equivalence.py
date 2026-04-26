"""h10 equivalence test: Grad-CAM combine formula.

The full pipeline (pretrained ResNet18 + hooks + heatmap overlay) downloads
weights and is matplotlib-heavy. We test the final Grad-CAM combine: given
the same activations and gradients, both implementations compute the same
heatmap.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close


def main():
    rng = np.random.default_rng(0)
    # PyTorch convention: (B, C, H, W); JAX (NHWC): (B, H, W, C).
    activations_chw = rng.standard_normal((1, 32, 7, 7)).astype(np.float32)
    grads_chw = rng.standard_normal((1, 32, 7, 7)).astype(np.float32) * 0.1
    activations_hwc = np.transpose(activations_chw, (0, 2, 3, 1))
    grads_hwc = np.transpose(grads_chw, (0, 2, 3, 1))

    # PyTorch: weights = grads.mean(dim=[2,3], keepdim=True); heatmap = (w*a).sum(1).relu().
    w_pt = torch.from_numpy(grads_chw).mean(dim=[2, 3], keepdim=True)
    h_pt = ((w_pt * torch.from_numpy(activations_chw)).sum(dim=1)
            .squeeze().relu()).numpy()
    h_pt = h_pt / (h_pt.max() + 1e-8)

    w_jx = jnp.mean(jnp.asarray(grads_hwc), axis=(1, 2), keepdims=True)
    h_jx = jnp.sum(w_jx * jnp.asarray(activations_hwc), axis=-1).squeeze()
    import jax
    h_jx = jax.nn.relu(h_jx)
    h_jx = h_jx / (jnp.max(h_jx) + 1e-8)
    assert_close(h_pt, np.asarray(h_jx), atol=1e-5, name="grad_cam_heatmap")
    print("[h10] PASS")


if __name__ == "__main__":
    main()
