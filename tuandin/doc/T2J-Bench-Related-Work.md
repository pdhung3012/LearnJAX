# Related work: data creation for LLM code translation

Literature review synthesizing prior work on data creation, evaluation, and
training strategies for LLM-based code translation and bug fixing, with
explicit mapping onto the T2J-Bench project plan.

Compiled: 2026-08-15.

## 1 · Timeline of the field

| Era | Representative works | Guiding hypothesis |
|---|---|---|
| **2020–2021** — unsupervised | TransCoder, TransCoder-ST | Monolingual code corpora + back-translation + DAE can substitute for parallel data. Add automated unit tests to filter invalid back-translations. |
| **2022–2023** — LLM instruction era | Self-Instruct, WizardCoder / Evol-Instruct, Magicoder / OSS-Instruct | Synthetic instruction data works if you seed generation from real code snippets to prevent LLM-hallucinated distributions. |
| **2023–2024** — execution-feedback | CoTran, UniTrans | Text generation is not enough; add an iterative repair loop guided by compilation errors and test-case failures. |
| **2024** — multi-agent + fine-grained | TransAgent, CodeRosetta | Decompose "fix" into specialized agents. Localize the specific broken block via execution alignment instead of retrying the whole file. |
| **2025–2026** — synthetic bugs + RL | SWE-Synth, SWE-Smith, SWE-Gym, EffiReasonTrans, Sol-Ver, DeepSWE, SWE-RL, Loong | Scale via targeted synthetic bug injection verified by execution. Train with RL against verifiable rewards. |
| **Parallel thread** — contamination | LiveCodeBench, contamination surveys | HumanEval/MBPP are contaminated. Use post-cutoff, time-segmented evaluation. |

Our prior T2J work (arXiv 2510.09898) sits in the 2024 era: iterative
human-in-the-loop repair on TorchLeet + CodeParrot with three custom
metrics.

## 2 · Six principal problems, tackled

### Problem 1 — Parallel data is scarce (especially for framework translation)

| Approach | How | Applicable to T2J? |
|---|---|---|
| Back-translation (TransCoder) | Train encoder on monolingual PT + JAX; back-translate | Weak: JAX corpus is ~100× smaller than PyTorch |
| Seed from real snippets (Magicoder / OSS-Instruct) | LLM invents problems anchored on random code | Directly applicable |
| Bug injection (SWE-Synth) | LLM rewrites correct code to introduce controllable faults | **Most promising** — bugs verifiable via existing oracle |
| Teacher distillation (EffiReasonTrans) | Strong reasoning model (DeepSeek-R1) generates triplets | Directly applicable |

**Consensus:** don't wait for natural parallel data. Manufacture it, and
verify with execution.

### Problem 2 — Text similarity ≠ semantic correctness

| Approach | Verification signal |
|---|---|
| TransCoder-ST | EvoSuite-generated unit tests |
| UniTrans | LLM-generated test cases, executed |
| CoTran | Compiler + symbolic execution |
| CodeRosetta | Run kernel, compare output (93% functional match reported) |
| EffiReasonTrans | Test-case pass rate as GRPO reward |
| SWE-Bench / SWE-Gym family | Real project test suites |

**Consensus:** BLEU/CodeBLEU are dead as correctness measures for
translation. Every serious 2024+ paper uses execution.

### Problem 3 — Scraped bug-fix data is noisy and unverifiable

- Traditional keyword mining (`fix|bug|repair`) yields ~50% noise.
- SWE-Synth showed synthetic bugs are indistinguishable from real GitHub
  bugs at ~55% human accuracy (near-random) — but *unlike* real bugs
  they come with a passing test, so they are verifiable.
- SWE-Smith scaled this to **50K instances across 128 projects** by
  perturbation.

**Consensus:** synthetic bugs with executable oracles beat noisy mined
data.

### Problem 4 — Distillation from strong to weak

- Standard recipe: strong teacher generates traces; weak student SFTs.
- Recent finding ("Strong Teacher Not Needed?"): stronger teachers don't
  always help — task-relevance of teacher output matters more than raw
  capability.
- EffiReasonTrans concrete recipe: DeepSeek-R1 synthesizes
  reasoning-annotated `(source, reasoning, target)` triplets → filter by
  execution → SFT → GRPO with execution reward. Reported **+27.4% CA**
  on Java → Python.

### Problem 5 — Whole-file repair is inefficient

TransAgent decomposes fixing into 4 specialized agents:

1. Initial translator
2. Syntax fixer
3. Code aligner (localizes divergence between source and target
   execution)
4. Semantic fixer

Beats UniTrans on both effectiveness and cost by narrowing the fixing
space per agent.

### Problem 6 — Contamination and generalization

- HumanEval, MBPP: heavily contaminated across all frontier models.
- LiveCodeBench solution: only score problems released *after* the
  model's cutoff date.
- Contamination surveys recommend: (a) time-segmented held-out sets,
  (b) never publish gold solutions publicly, (c) report contamination
  diagnostics per model.

## 3 · The recipe that emerges (2026 consensus)

Six invariants across every serious paper:

1. **Executable oracle is the ground truth.** LLM judgments, similarity
   scores → calibration only.
2. **Iterative refinement > single-shot.** 2–5 fix rounds is standard.
3. **Bug/fix trajectories carry more supervision signal than end-state
   pairs.** SFT on trajectories, not just final answers.
4. **Diversify via source seeding OR controlled bug injection.** Never
   trust a single LLM's failure distribution alone.
5. **Verifiable reward = RL fuel.** GRPO/DPO on oracle-based rewards is
   the post-DeepSeek-R1 recipe.
6. **Held-out = temporal, not random.** Problems released after model
   cutoffs.

## 4 · Integrated plan for T2J-Bench

Each recommendation maps to a specific finding in §2/§3. Additions are
marked **NEW**; replacements are marked **REPLACES**.

### 4A · Data construction (Phase 1–2)

| Piece | Current plan | Recommended update |
|---|---|---|
| Source problems | TorchLeet 20 kernels | **NEW**: add SWE-Synth-style bug injection over the existing 40 gold cases → target ~2,000 verifiable bug/fix pairs from 40 seeds. Cheap to scale. |
| Bug source | Weak-LLM translation errors only | **NEW**: also inject controlled bugs (dtype swaps, axis swaps, missing `use_running_average`, PRNG-key misuse, JAX-specific tracer errors). Yields a taxonomy figure for the paper. |
| Verification | Executable oracle (built) | Keep — already ahead of most 2024 papers here |
| Held-out set | Same 20 kernels | **REPLACES**: build a post-2025 held-out set from GitHub PyTorch code (LiveCodeBench methodology). Report on it separately from TorchLeet. |
| Multi-source seeding | Not currently done | **NEW**: OSS-Instruct pattern — sample real PyTorch snippets from HuggingFace `transformers` sub-modules whose JAX equivalents post-date model cutoffs. |

### 4B · Pipeline architecture

| Piece | Current | Update |
|---|---|---|
| Roles | Translator → junior fixer → senior fixer | **NEW**: adopt TransAgent decomposition. Add a "code aligner" agent that runs a diff-of-execution between PyTorch and JAX (both write to a trace log) to localize the divergent block, plus a "shape fixer" specialized on Flax dimension conventions. |
| Reward signal | Pass/fail only | **NEW**: surface fine-grained per-tensor / per-layer diffs into the next fix prompt. `oracle.per_output` already collects this. |

### 4C · Training strategy (Phase 3)

Follow the **EffiReasonTrans** recipe, adapted to PT→JAX:

1. **Teacher**: DeepSeek V4 Pro (cheap) or Claude Opus 4.8 (higher
   quality) generates `(pytorch, reasoning_trace, jax)` triplets with
   per-step justification.
2. **Filter**: keep only triplets that pass the executable oracle.
3. **SFT**: fine-tune Qwen2.5-Coder-7B on filtered triplets. LoRA rank
   32 as starting point.
4. **RL**: GRPO with reward = oracle pass rate (binary) + a small
   penalty for exceeding a target token budget. Borrow EffiReasonTrans
   hyperparameters.

Expected win: EffiReasonTrans got +27.4% Java→Python. Similar delta for
PT→JAX would put a 7B model at Claude-Opus-adjacent performance.

### 4D · Evaluation (Phase 4 headline table)

Baselines required for a top venue:

- `torch2jax` library (already in `tuandin/baselines/t2j/`)
- Zero-shot Claude Opus 4.8, GPT-5.6 Sol Pro, Gemini 3.1 Pro
- **UniTrans-style** prompting (test-case-guided iteration) with
  off-the-shelf strong model
- **TransAgent-style** multi-agent (4 specialized agents) with
  off-the-shelf strong model
- Our ICL-augmented Qwen 2.5 Coder 7B
- Our SFT+GRPO Qwen 2.5 Coder 7B (headline number)

Metrics:

- Drop RC/RTC/LMC from the correctness gate → diagnostics only
- Report pass@1, pass@5, avg fix rounds (standard)
- Split results: TorchLeet (contaminated), held-out post-cutoff (clean),
  cross-target (Triton via KernelBench)

### 4E · Cross-target generalization (novel angle)

Almost no existing paper trains for cross-target translation. If our
PT→JAX fine-tuned model *also* improves PT→Triton (via
`tuandin/kernelbench_cnn/`) without task-specific training, that is a
transfer result — a fresh contribution independent of scale.

### 4F · What to explicitly NOT do

- Don't scale the human loop — 2025 papers show agent pipelines +
  executable oracles match human quality on much bigger scales
  (SWE-Synth's Turing-test result).
- Don't add TF↔PT — the community is moving away from TF; framing
  suffers.
- Don't rely on LLM-as-judge for correctness — only for readability, and
  calibrated against a human sample.
- Don't add more custom metrics (`T2J_CodeTrans_Score`, etc.). Reviewers
  strongly prefer standard pass@k over paper-specific metrics.

## 5 · Paper positioning

The differentiator becomes: **"first bug/fix benchmark for a
domain-specific ML framework translation, built entirely via agent
pipeline + executable oracle, with contamination-resistant temporal
split and cross-target transfer results."**

That story fits NeurIPS Datasets & Benchmarks or an ICSE / FSE tools
paper. Each of TransAgent (FSE 2026), SWE-Synth (ICSE 2026
Distinguished), SWE-Gym (ICML 2025), and LiveCodeBench (ICLR 2025) is a
comparable landing to aim at with this framing.

## References

- **Learning Bug Context for PyTorch-to-JAX Translation with LLMs** (our
  prior work). arXiv 2510.09898.
  https://arxiv.org/html/2510.09898v1
- **T2J** (OpenReview forum).
  https://openreview.net/forum?id=dZvzVVa1Qh
- **TransCoder-ST — Leveraging Automated Unit Tests for Unsupervised
  Code Translation**. arXiv 2110.06773.
  https://arxiv.org/pdf/2110.06773
- **ExeCoder: Empowering LLMs with Executability Representation for Code
  Translation**. arXiv 2501.18460.
  https://arxiv.org/html/2501.18460v2
- **CodeRosetta: Unsupervised Code Translation for Parallel Programming**
  (NeurIPS 2024).
  https://openreview.net/forum?id=V6hrg4O9gg
- **UniTrans / Exploring and Unleashing the Power of LLMs in Automated
  Code Translation** (FSE 2024). arXiv 2404.14646.
  https://arxiv.org/abs/2404.14646
- **TransAgent: Fine-Grained Execution Alignment for Code Translation**
  (FSE 2026). arXiv 2409.19894.
  https://arxiv.org/html/2409.19894
- **CoTran: LLM Code Translator with RL from Compiler + Symbolic
  Execution**. arXiv 2306.06755.
  https://arxiv.org/pdf/2306.06755
- **EffiReasonTrans: RL-Optimized Reasoning for Code Translation**.
  arXiv 2510.18863.
  https://arxiv.org/html/2510.18863
- **Magicoder: Empowering Code Generation with OSS-Instruct** (ICML
  2024). arXiv 2312.02120.
  https://arxiv.org/abs/2312.02120
- **SWE-Synth: Synthesizing Verifiable Bug-Fix Data** (ICSE 2026
  Distinguished). arXiv 2504.14757.
  https://arxiv.org/html/2504.14757
- **SWE-Smith: Scaling Data for SWE-agents** (NeurIPS 2025 D&B
  Spotlight).
  https://github.com/SWE-bench/SWE-smith
- **SWE-Gym: Training Real-World SWE Agents** (ICML 2025).
  https://github.com/SWE-Gym/SWE-Gym
- **Sol-Ver: Self-Play Solver-Verifier Framework**. arXiv 2502.14948.
  https://arxiv.org/abs/2502.14948v2
- **SWE-RL: RL for Software Engineering Tasks**.
  https://medium.com/@techsachin/swe-rl-approach-to-scale-reinforcement-learning-based-llm-reasoning-for-software-engineering-tasks-883fa1a9c5a9
- **Strong Teacher Not Needed? On Distillation in LLM Pretraining**.
  arXiv 2605.23857.
  https://arxiv.org/abs/2605.23857
- **LiveCodeBench: Holistic and Contamination-Free Evaluation** (ICLR
  2025).
  https://proceedings.iclr.cc/paper_files/paper/2025/file/94074dd5a072d28ff75a76dabed43767-Paper-Conference.pdf
- **A Survey on Data Contamination for LLMs**. arXiv 2502.14425.
  https://arxiv.org/html/2502.14425v2
- **Enhancing LLMs in Long Code Translation through Instrumentation and
  Program State Alignment**. arXiv 2504.02017.
  https://arxiv.org/pdf/2504.02017
