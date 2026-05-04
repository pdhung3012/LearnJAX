# `torch2jax` produces no JAX source file

Unlike a source-to-source translator (or a cheap LLM that emits
`jax_code.py`), `torch2jax` works by *abstract interpretation* — it
intercepts `torch.*` calls at runtime and re-dispatches them to JAX.

The "translation" exists only as the runtime trace and never as a
source file. The conceptual equivalent of `jax_code.py` here is the
PyTorch source itself plus the `torch2jax.t2j_module(pt_model)` wrapping
in `../../adapter.py`.

If/when this case is fed to a *cheap LLM* baseline (next phase), this
NOTE.md will be replaced with the actual `jax_code.py` source the LLM
emitted.
