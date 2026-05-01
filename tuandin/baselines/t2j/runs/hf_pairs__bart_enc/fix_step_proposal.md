# Proposed fix step for `hf_pairs/bart_enc`

**Status:** RUNTIME_ERROR

**Error category:** `missing_op:torch.nn.functional.embedding`

**Proposal:** Algorithmic fix: replace `torch.nn.functional.embedding(...)` with a JAX-native equivalent in the jax forward, OR (upstream) register a `@torch2jax.implements(torch.nn.functional.embedding)` rule that dispatches to the corresponding `jax.numpy` / `jax.lax` op.

_Note: torch2jax is deterministic, so this proposal is documentation only — no fix is automatically applied. The actual fix-step iteration loop (captured in `fix_steps/`) only applies to LLM baselines._
