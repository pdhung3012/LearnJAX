# Proposed fix step for `hf_pairs/t5_enc`

**Status:** RUNTIME_ERROR

**Error category:** `unhashable`

**Proposal:** Algorithmic fix: avoid using the traced tensor as a dict key (replace with an int/string identifier, or rewrite the indexing pattern).

_Note: torch2jax is deterministic, so this proposal is documentation only — no fix is automatically applied. The actual fix-step iteration loop (captured in `fix_steps/`) only applies to LLM baselines._
