# T2J-Bench: Agent-Pipeline Implementation Plan

Working plan for scaling the PyTorch→JAX bug/fix dataset via a full agent
pipeline (no human in the loop for CC/TC/LMC; executable oracle only).

## 1. Model roster

All slugs are current OpenRouter IDs. Prices are USD per 1M tokens
(input / output). Snapshot: 2026-08-15.

### 1.1 Weak translators (open-weight, ~3B–15B, fine-tuneable)

| Model | Router slug | Params | $/M in | $/M out | Ctx |
|---|---|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | `qwen/qwen-2.5-7b-instruct` | 7B | 0.10 | 0.20 | 32K |
| Qwen3-8B | `qwen/qwen3-8b` | 8B | 0.12 | 0.46 | 131K |
| Qwen3-14B | `qwen/qwen3-14b` | 14B | 0.12 | 0.24 | 131K |
| Qwen3-Coder-Flash | `qwen/qwen3-coder-flash` | small MoE | 0.20 | 0.98 | 1M |
| Codestral-2508 | `mistralai/codestral-2508` | 22B | 0.30 | 0.90 | 256K |
| Ministral-8B | `mistralai/ministral-8b-2512` | 8B | 0.15 | 0.15 | 262K |
| Granite-4.1-8B | `ibm-granite/granite-4.1-8b` | 8B | 0.05 | 0.10 | 131K |
| Phi-4 | `microsoft/phi-4` | 14B | 0.07 | 0.14 | 16K |
| Gemma-3-12B-it | `google/gemma-3-12b-it` | 12B | 0.05 | 0.15 | 131K |
| Llama-3.1-8B-Instruct | `meta-llama/llama-3.1-8b-instruct` | 8B | 0.05 | 0.08 | 131K |
| gpt-oss-20B | `openai/gpt-oss-20b` | 20B MoE | 0.03 | 0.13 | 131K |

### 1.2 Strong fixers (frontier, senior role)

| Model | Router slug | Family | $/M in | $/M out | Ctx |
|---|---|---|---:|---:|---:|
| Claude Opus 4.8 | `anthropic/claude-opus-4.8` | Anthropic | 5.00 | 25.00 | 1M |
| Claude Sonnet 4.6 | `anthropic/claude-sonnet-4.6` | Anthropic | 3.00 | 15.00 | 1M |
| GPT-5.4 | `openai/gpt-5.4` | OpenAI | 2.50 | 15.00 | 1M |
| GPT-5.6 Sol Pro | `openai/gpt-5.6-sol-pro` | OpenAI | 5.00 | 30.00 | 1M |
| o3 | `openai/o3` | OpenAI | 2.00 | 8.00 | 200K |
| Gemini 3.1 Pro | `google/gemini-3.1-pro-preview` | Google | 2.00 | 12.00 | 1M |
| Gemini 2.5 Pro | `google/gemini-2.5-pro` | Google | 1.25 | 10.00 | 1M |
| Grok 4.6 | `x-ai/grok-4.6` | xAI | 2.00 | 6.00 | 500K |
| DeepSeek V4 Pro | `deepseek/deepseek-v4-pro` | DeepSeek | 1.17 | 2.34 | 1M |
| GLM-5.2 | `z-ai/glm-5.2` | Zhipu | 0.46 | 1.45 | 1M |
| Kimi K3 | `moonshotai/kimi-k3` | Moonshot | 3.00 | 15.00 | 1M |
| Kimi K2.7-Code | `moonshotai/kimi-k2.7-code` | Moonshot | 0.71 | 3.50 | 262K |

### 1.3 Judges (for RC only — third family, independence constraint)

| Model | Router slug | $/M in | $/M out |
|---|---|---:|---:|
| Gemini 2.5 Flash | `google/gemini-2.5-flash` | 0.30 | 2.50 |
| GPT-5-mini | `openai/gpt-5-mini` | 0.25 | 2.00 |
| Gemini 3.5 Flash | `google/gemini-3.5-flash` | 1.50 | 9.00 |
| GPT-5.4-mini | `openai/gpt-5.4-mini` | 0.75 | 4.50 |

## 2. Cost tiers

Two shared configurations. Colleague runs the cheap tier for scale;
principal (Tuan) runs the premium tier for headline numbers.

### Cheap tier (colleague default)

Optimized for cost. Total spend for a 10K-item dataset ≈ **$100–200**.

- Weak translators: `granite-4.1-8b`, `llama-3.1-8b-instruct`,
  `gpt-oss-20b`, `phi-4`, `gemma-3-12b-it` (all ≤ $0.15/M output)
- Fixer (junior): `z-ai/glm-5.2` ($0.46/$1.45)
- Fixer (senior escalation): `deepseek/deepseek-v4-pro` ($1.17/$2.34)
- Judge (RC only): `google/gemini-2.5-flash` ($0.30/$2.50)

### Premium tier (principal)

Optimized for headline results and hard-case coverage.
Total spend for a 10K-item dataset ≈ **$1.5K–3K**.

- Weak translators: same panel + `codestral-2508`, `qwen3-coder-flash`
  (adds specialized code models)
- Fixer (junior): `anthropic/claude-sonnet-4.6`
- Fixer (senior escalation): `anthropic/claude-opus-4.8`
- Judge (RC only): `google/gemini-3.5-flash` (different family from
  Anthropic fixers)

### Independence constraint

For no circular validation, the four roles below must be from
different vendor families:

- Translator (weak) — Alibaba (Qwen), Meta (Llama), IBM (Granite), etc.
- Fixer (junior) — Anthropic OR Zhipu
- Fixer (senior) — Anthropic OR DeepSeek
- Judge (RC) — Google (Gemini)

Objective criteria (CC, TC, LMC, RTC) are judged by the executable
oracle, not an LLM.

## 3. Pipeline architecture

```
PyTorch source (from existing tuandin/jax_translations/*/pytorch_code.py)
  │
  ▼
[Translator: weak LLM]  ──► candidate jax_code.py (v0)
  │
  ▼
[Oracle: executable]
  │  runs inputs.npz through jax_code.compute(), compares to expected.npz
  │  → CC (imports/compiles), TC (numerical match)
  │
  ├── PASS ────► log as "trivial for this translator", skip (not useful data)
  │
  └── FAIL ────► capture error message
        │
        ▼
     [Fixer: junior LLM] × up to N rounds
        │  input = (pytorch_code, jax_code_current, oracle_error)
        │  output = jax_code (v+1)
        │  each intermediate (bug, fix) pair logged to dataset
        │
        ├── PASS at round k ───► log final fix as "junior-solvable"
        │
        └── FAIL after N ─────► escalate
              │
              ▼
           [Fixer: senior LLM] × up to M rounds
              │
              ├── PASS ────► log final fix as "senior-required"
              │
              └── FAIL ────► log as "unresolved", set aside for
                             later human review or dataset filtering
```

### Oracle contract

Every case has a `compute(inputs: dict) -> dict` contract, with paired
`inputs.npz` and `expected.npz`. This is already established across
`tuandin/jax_translations/`, `tuandin/hf_pairs/`, `tuandin/kernelbench_cnn/`,
and `tuandin/opinfo/`. The oracle is deterministic Python, not an LLM.

### RC judge (readability)

Runs only after the executable oracle passes. LLM-as-judge scores whether
`jax_code.py` uses idiomatic JAX/Flax abstractions and faithfully mirrors
the PyTorch structure. Calibrate on 50 human-labeled samples, report
Cohen's κ before trusting the score.

### RTC (runtime)

Measured separately with JIT warmup + `block_until_ready`. **Not part of
correctness gate** — reported as a diagnostic metric.

## 4. Cost estimates

Assumptions per pipeline call: ~2K input + 2K output tokens.

| Stage | Cheap tier | Premium tier |
|---|---:|---:|
| 8 weak translators × 10K calls | ~$40 | ~$100 |
| Junior fixer, ~3 rounds avg × 7K failures | ~$40 | ~$500 |
| Senior escalation × 30% of failures | ~$50 | ~$1,500 |
| RC judge × 10K | ~$15 | ~$50 |
| **Total (10K items)** | **~$150** | **~$2,150** |

Batching (OpenRouter `:batch` variants) cuts input costs ~50% for both tiers.

## 5. Implementation plan

### Phase 0 — prototype (this repo, `tuandin/t2j_bench/`)

Runnable end-to-end on 1 existing case from `jax_translations/`.

- [x] Plan doc (this file)
- [ ] `models.py` — roster + tier defs
- [ ] `openrouter_client.py` — thin wrapper, key from `open_router.txt`
- [ ] `oracle.py` — plugs into existing contract
- [ ] `pipeline.py` — translate/fix loop with logging
- [ ] `prompts.py` — translator + fixer prompts
- [ ] `run.py` — CLI, `--tier cheap|premium`, `--case <path>`
- [ ] `README.md` — quickstart

### Phase 1 — scale on existing 40 cases

- Run cheap tier across `jax_translations/*` + `hf_pairs/*` (~40 cases,
  ~40 weak-model × 40 case = 1.6K translations)
- Collect bug/fix trajectories in JSONL
- Verify oracle failure categorization is clean

### Phase 2 — expand corpus

- Add TorchLeet (20 kernels) and KernelBench relevant tier
- Target 500+ source problems, held-out train/test split at the
  problem level
- Human-verify 50-sample calibration set for RC judge

### Phase 3 — fine-tune weak models

Not on OpenRouter (inference-only). Use HuggingFace weights + rented
GPUs (Modal / Together / RunPod).

- Baseline: zero-shot pass rates for each weak model
- LoRA fine-tune Qwen2.5-Coder-7B on collected bug/fix pairs
- Evaluate on held-out problems, report pass@1

### Phase 4 — headline table

Baselines the reviewers will demand:

- `torch2jax` library (already in `baselines/t2j/`, pass@1=6.7%)
- Zero-shot Claude Opus 4.8 direct translation
- Zero-shot GPT-5.6 Sol Pro direct translation
- Zero-shot best open model
- Fine-tuned Qwen2.5-Coder-7B
- ICL-augmented Qwen2.5-Coder-7B

## 6. Division of labor

**Colleague (cheap tier):**
- Runs Phase 1 across full weak-model panel with GLM-5.2 fixer
- Owns the JSONL dataset assembly
- Runs Phase 2 corpus expansion

**Principal (premium tier):**
- Runs premium fixer on the hardest ~20% of cases (Claude-quality
  fixes for the paper's "hard problems" analysis)
- Owns human calibration set for RC judge
- Owns fine-tuning experiments (Phase 3)

The `--tier` CLI flag switches configurations. Both write to the same
JSONL schema so results merge.

## 7. Open questions

- Fine-tune target: base or instruct? LoRA rank? — decide before Phase 3
- Held-out split protocol: by problem, by kernel-family, by both? —
  decide before Phase 2
- Second target framework (Triton kernels? ONNX?) — deferred; see
  paper critique for rationale
