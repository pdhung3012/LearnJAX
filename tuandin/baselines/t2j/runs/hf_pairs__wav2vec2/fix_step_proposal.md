# Proposed fix step for `hf_pairs/wav2vec2`

**Status:** RUNTIME_ERROR

**Error category:** `missing_attr:unsqueeze`

**Proposal:** Algorithmic fix: replace `tensor.unsqueeze` access with the JAX-functional equivalent (e.g., shape/device queries become Python-side metadata, `.long()` becomes an explicit `astype(int64)` cast, etc.).

_Note: torch2jax is deterministic, so this proposal is documentation only — no fix is automatically applied. The actual fix-step iteration loop (captured in `fix_steps/`) only applies to LLM baselines._
