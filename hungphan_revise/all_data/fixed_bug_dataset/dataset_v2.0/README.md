## Two-tier equivalence check

Tier 1 — Unit equivalence (tolerance 1e-5): given identical hardcoded tensors injected into both frameworks, the deterministic functions must produce numerically matching outputs. Three tests:

- test_forward_pass(): same output given identical X and params
- test_loss(): same MSE loss given identical inputs
- test_train_step(): same updated params and loss after one gradient step

Tier 2 — Convergence equivalence (tolerance 1e-2): after full training from identical starting params and data, learned parameters and predictions agree. One test:

- test_convergence(): calls train_model(X, y, params) directly with hardcoded tensors

## Mock injection strategy

Rather than modifying the source files or using pytest mock, hardcoded numpy tensors (X_np, y_np, W_np, B_np, X_test_np) are defined once at the top of the test file and shared across all tests. PyTorch's random init is bypassed via .weight.data and .bias.data. JAX's random init is bypassed by constructing the params dict directly from the same numpy arrays. No RNG is called anywhere in the test file.

## What is and isn't tested for numerical equivalence

RNG functions (generate_data, init_params) are not tested for numerical equivalence — JAX and PyTorch produce different values from the same seed by design. Only deterministic functions are tested in Tier 1. Tier 2 validates end-to-end behaviour despite the RNG difference, using a looser tolerance to account for 1000 steps of accumulated float32 differences.

## Verifier file rules

Imports functions from jax_code_fixed.py directly — never copies code into the test file
All three files (pytorch_code.py, jax_code_fixed.py, test_equivalence.py) must live in the same directory and always be committed together