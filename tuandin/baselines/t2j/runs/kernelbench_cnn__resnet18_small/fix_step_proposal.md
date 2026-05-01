# Proposed fix step for `kernelbench_cnn/resnet18_small`

**Status:** RUNTIME_ERROR

**Error category:** `missing_kwarg:dim`

**Proposal:** Algorithmic fix: change call site to use the positional/alternative form that does not pass `dim=...`, or (upstream) extend the torch2jax interceptor to forward the kwarg.

_Note: torch2jax is deterministic, so this proposal is documentation only — no fix is automatically applied. The actual fix-step iteration loop (captured in `fix_steps/`) only applies to LLM baselines._
