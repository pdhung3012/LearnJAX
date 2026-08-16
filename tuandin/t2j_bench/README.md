# T2J-Bench — OpenRouter setup and model recommendations

This directory ships two things:

1. **`demo.py`** — a zero-dependency, standalone script that shows how to
   authenticate with OpenRouter and call the three pipeline roles (weak
   translator, strong fixer, RC judge). Adapt it however you like.
2. **The full pipeline** (`run.py`, `pipeline.py`, `oracle.py`) — used for
   large-scale bug/fix data collection. See `## Full pipeline` below.

## 1 · Quickstart (demo)

Pass the OpenRouter key via env var. The demo has no dependencies beyond
the Python standard library.

```bash
OPEN_ROUTER_KEY=sk-or-... python -m tuandin.t2j_bench.demo
```

Expected output (~3s, ~$0.0006 total):

```
--- WEAK translator · ibm-granite/granite-4.1-8b ($0.05/$0.10 per 1M) ---
latency: 0.95s | in=106 out=69 tokens | cost=$0.000012
...
--- STRONG fixer · z-ai/glm-5.2 ($0.46/$1.45 per 1M) ---
latency: 1.63s | in=211 out=78 tokens | cost=$0.000210
...
--- RC judge · google/gemini-2.5-flash ($0.30/$2.50 per 1M) ---
latency: 1.47s | in=200 out=112 tokens | cost=$0.000340
...
=== total demo cost: $0.000562 ===
```

If you'd rather keep the key in a file, drop it into
`tuandin/open_router.txt` and the full pipeline will pick it up
automatically (the demo is env-var only).

## 2 · Model roster

Prices are USD per 1M tokens (input / output). Snapshot: 2026-08-15.
Re-check `openrouter.ai/models` before large runs — pricing changes.

### Weak translators (open-weight, fine-tuneable)

Used to generate buggy JAX for supervision. Must be open-weight if you
plan to fine-tune later. All are ≤ 22B params.

| Model | Router slug | Params | $/M in | $/M out |
|---|---|---:|---:|---:|
| Granite-4.1-8B | `ibm-granite/granite-4.1-8b` | 8B | 0.05 | 0.10 |
| Llama-3.1-8B-Instruct | `meta-llama/llama-3.1-8b-instruct` | 8B | 0.05 | 0.08 |
| Gemma-3-12B-it | `google/gemma-3-12b-it` | 12B | 0.05 | 0.15 |
| Phi-4 | `microsoft/phi-4` | 14B | 0.07 | 0.14 |
| gpt-oss-20B | `openai/gpt-oss-20b` | 20B MoE | 0.03 | 0.13 |
| Qwen2.5-7B-Instruct | `qwen/qwen-2.5-7b-instruct` | 7B | 0.10 | 0.20 |
| Qwen3-8B | `qwen/qwen3-8b` | 8B | 0.12 | 0.46 |
| Qwen3-14B | `qwen/qwen3-14b` | 14B | 0.12 | 0.24 |
| Qwen3-Coder-Flash | `qwen/qwen3-coder-flash` | small MoE | 0.20 | 0.98 |
| Ministral-8B | `mistralai/ministral-8b-2512` | 8B | 0.15 | 0.15 |
| Codestral-2508 | `mistralai/codestral-2508` | 22B | 0.30 | 0.90 |

### Strong fixers (frontier)

Used to repair buggy JAX. Ordered cheap → expensive.

| Model | Router slug | Family | $/M in | $/M out |
|---|---|---|---:|---:|
| GLM-5.2 | `z-ai/glm-5.2` | Zhipu | 0.46 | 1.45 |
| Kimi K2.7-Code | `moonshotai/kimi-k2.7-code` | Moonshot | 0.71 | 3.50 |
| DeepSeek V4 Pro | `deepseek/deepseek-v4-pro` | DeepSeek | 1.17 | 2.34 |
| Gemini 2.5 Pro | `google/gemini-2.5-pro` | Google | 1.25 | 10.00 |
| Grok 4.6 | `x-ai/grok-4.6` | xAI | 2.00 | 6.00 |
| o3 | `openai/o3` | OpenAI | 2.00 | 8.00 |
| Gemini 3.1 Pro | `google/gemini-3.1-pro-preview` | Google | 2.00 | 12.00 |
| GPT-5.4 | `openai/gpt-5.4` | OpenAI | 2.50 | 15.00 |
| Claude Sonnet 4.6 | `anthropic/claude-sonnet-4.6` | Anthropic | 3.00 | 15.00 |
| Kimi K3 | `moonshotai/kimi-k3` | Moonshot | 3.00 | 15.00 |
| Claude Opus 4.8 | `anthropic/claude-opus-4.8` | Anthropic | 5.00 | 25.00 |
| GPT-5.6 Sol Pro | `openai/gpt-5.6-sol-pro` | OpenAI | 5.00 | 30.00 |

### Judges (RC / readability only)

| Model | Router slug | $/M in | $/M out |
|---|---|---:|---:|
| GPT-5-mini | `openai/gpt-5-mini` | 0.25 | 2.00 |
| Gemini 2.5 Flash | `google/gemini-2.5-flash` | 0.30 | 2.50 |
| GPT-5.4-mini | `openai/gpt-5.4-mini` | 0.75 | 4.50 |
| Gemini 3.5 Flash | `google/gemini-3.5-flash` | 1.50 | 9.00 |

## 3 · Recommended tiers

Two configurations we've aligned on:

### Cheap tier (default for scale runs)

- Weak translators: Granite-8B, Llama-8B, Gemma-12B, Phi-4, gpt-oss-20B
  (all ≤ $0.15/M out)
- Junior fixer: **GLM-5.2**
- Senior fixer: **DeepSeek V4 Pro**
- RC judge: **Gemini 2.5 Flash**

Total cost per case × translator × 5 fix rounds: **~$0.03**. See scale
table below.

### Premium tier (for principal-run headline numbers)

- Weak translators: cheap panel + Qwen2.5-Coder-7B + Codestral +
  Qwen3-Coder-Flash
- Junior fixer: **Claude Sonnet 4.6**
- Senior fixer: **Claude Opus 4.8**
- RC judge: **Gemini 3.5 Flash** (different family from fixers — see
  independence constraint)

Total cost per case × translator × 5 fix rounds: **~$0.30**.

## 4 · Cost estimates at scale

Assumes ~2K in + 2K out tokens per LLM call, and worst-case round budget
(1 translate + 3 junior + 2 senior = 6 calls per (case × translator)).
Real spend will be lower because many pairs terminate early (`TRIVIAL`
after round 0, `PASS` after 1–2 fix rounds).

| Scale | Cheap tier | Premium tier |
|---|---:|---:|
| Demo (3 calls) | $0.0006 | $0.005 |
| 1 case × 5 translators (e.g. `--case e1`) | $0.15 | $1.50 |
| 40 cases × 5 translators (existing `jax_translations/`) | $6 | $60 |
| 100 cases × 8 translators | $24 | $240 |
| 1,000 cases × 8 translators | $240 | $2,400 |
| 10,000 cases × 8 translators (paper-scale corpus) | $2,400 | $24,000 |

Real observed rates from e1 validation run: **$0.0016 for 5 translators
on the cheap tier** (most cases were TRIVIAL after round 0). Assume 5–10×
lower than the worst-case table above for empirical planning.

## 5 · Credit alignment guidance

- **Anything up to ~$50/day: no pre-check needed.** Small case sets, ICL
  ablations, prompt tuning — just run it.
- **$50 – $500 per run: notify before starting.** This is the "run the
  cheap tier across a new corpus tier" zone. A quick heads-up so we
  don't double-book credits.
- **>$500 per run: get explicit sign-off.** These are only Phase 3
  fine-tuning eval or full-corpus premium runs. Confirm with Tuan first
  and budget the credit allocation.

Rule of thumb: if you're about to run something >100× larger than the
demo, do the math with the table above, and if it lands over $50,
message before hitting go.

## 6 · Independence constraint (avoid circular validation)

For no LLM to judge its own family's output:

- Translator ≠ Fixer family (Alibaba/Meta/IBM ≠ Anthropic/OpenAI/etc.)
- Junior fixer ≠ Senior fixer family (ideally)
- Judge family ≠ Fixer family (use Google Gemini judge with Anthropic
  fixers, or vice versa)

Objective criteria (compilation, test-case, loss trajectory) are checked
by the executable oracle (Python subprocess against `inputs.npz` /
`expected.npz`), never by an LLM. Only the RC (readability) criterion
uses an LLM judge, and we calibrate it against a 50-sample
human-labeled set before trusting it in the paper.

The `check_independence()` helper in `models.py` warns at pipeline
startup if the configured tier violates this.

## 7 · Full pipeline (advanced)

For collecting bug/fix trajectories at scale, `run.py` drives the
translate → oracle → fix loop over any case that follows the
`compute(inputs: dict) -> dict` contract (see `tuandin/jax_translations/`
for 30 examples, `tuandin/hf_pairs/` for 10 more).

```bash
# Single case
python -m tuandin.t2j_bench.run --tier cheap \
    --case tuandin/jax_translations/e1

# Whole corpus, cheap tier
python -m tuandin.t2j_bench.run --tier cheap \
    --case-glob 'tuandin/jax_translations/*' \
    --out tuandin/t2j_bench/runs/cheap_full.jsonl
```

Output is JSONL, one record per `(case × translator)` pair, with the
full trajectory (each round's code, oracle status, error, cost, latency).

`final_status` values:
- `TRIVIAL` — weak translator solved zero-shot (no supervision data,
  filter out)
- `PASS` — required ≥ 1 fix round (this is the useful supervision data)
- `UNRESOLVED` — neither junior nor senior fixed it within round budget

See `tuandin/doc/T2J-Bench-Implementation-Plan.md` for the design
rationale, phases, and division of labor between principal and
collaborator.

## 8 · Known gaps in the prototype

- No RC judge in the pipeline yet (schema slot exists; wire it in when
  the loop is validated at scale)
- No RTC (runtime) measurement (should be a diagnostic, not a
  correctness gate)
- No `:batch` variant support (halves input cost for high-volume runs)
- No held-out train/test split enforcement — this is a corpus-layer
  concern, not per-case
