"""Run torch2jax baseline on every in-scope case and produce per-case
artifacts that mirror the planned fix-trajectory dataset structure.

For each case, this writes:
  runs/<tier>__<case>/
    pytorch_code.py        (copy of input the tool sees)
    gold_jax_code.py       (copy of our hand-written reference translation)
    candidate_jax/NOTE.md  (torch2jax produces no source — note explains)
    error.txt              (full traceback or "PASS")
    result.json            (status + max_diff + error_category)
    fix_step_proposal.md   (one-line description of what a fix would be)

Then writes a top-level summary: results.md (table) + manifest.json (machine-readable).
"""
import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
sys.path.insert(0, str(ROOT))
from adapter import make_compute

CASES = [
    ("hf_pairs",        "bert",                ("input_ids", "attention_mask"), ("last_hidden_state",)),
    ("hf_pairs",        "gpt2",                ("input_ids", "attention_mask"), ("last_hidden_state",)),
    ("hf_pairs",        "roberta",             ("input_ids", "attention_mask"), ("last_hidden_state",)),
    ("hf_pairs",        "distilbert",          ("input_ids", "attention_mask"), ("last_hidden_state",)),
    ("hf_pairs",        "albert",              ("input_ids", "attention_mask"), ("last_hidden_state",)),
    ("hf_pairs",        "t5_enc",              ("input_ids", "attention_mask"), ("last_hidden_state",)),
    ("hf_pairs",        "vit",                 ("pixel_values",),               ("last_hidden_state",)),
    ("hf_pairs",        "bart_enc",            ("input_ids", "attention_mask"), ("last_hidden_state",)),
    ("hf_pairs",        "mistral",             ("input_ids", "attention_mask"), ("last_hidden_state",)),
    ("hf_pairs",        "wav2vec2",            ("input_values",),               ("last_hidden_state",)),
    ("kernelbench_cnn", "simple_bn_block",     ("pixel_values",),               ("logits",)),
    ("kernelbench_cnn", "resnet18_small",      ("pixel_values",),               ("logits",)),
    ("kernelbench_cnn", "vgg_bn_small",        ("pixel_values",),               ("logits",)),
    ("kernelbench_cnn", "mobilenet_v2_small",  ("pixel_values",),               ("output",)),
    ("kernelbench_cnn", "effnet_mb_block",     ("pixel_values",),               ("output",)),
]

ATOL = 1e-5
RTOL = 1e-5

CANDIDATE_NOTE = """\
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
"""


def _categorize(error_text: str) -> str:
    """Extract a short category label from the error text."""
    if not error_text:
        return ""
    if "Unhandled function call" in error_text:
        # Pull out the torch op name from the typical traceback line.
        import re
        m = re.search(r"Unhandled function call: ([\w\.]+)\(", error_text)
        return f"missing_op:{m.group(1)}" if m else "missing_op"
    if "object has no attribute" in error_text:
        import re
        m = re.search(r"has no attribute '(\w+)'", error_text)
        return f"missing_attr:{m.group(1)}" if m else "missing_attr"
    if "got an unexpected keyword argument" in error_text:
        import re
        m = re.search(r"keyword argument '(\w+)'", error_text)
        return f"missing_kwarg:{m.group(1)}" if m else "missing_kwarg"
    if "unhashable type" in error_text:
        return "unhashable"
    return "other"


def _propose_fix(category: str, case: str) -> str:
    """One-line fix proposal based on the failure category. Not applied
    automatically — meant as documentation of what an algorithmic fix step
    would entail."""
    if category.startswith("missing_op:"):
        op = category.split(":", 1)[1]
        return (
            f"Algorithmic fix: replace `{op}(...)` with a JAX-native equivalent in the "
            f"jax forward, OR (upstream) register a `@torch2jax.implements({op})` rule "
            f"that dispatches to the corresponding `jax.numpy` / `jax.lax` op."
        )
    if category.startswith("missing_attr:"):
        attr = category.split(":", 1)[1]
        return (
            f"Algorithmic fix: replace `tensor.{attr}` access with the JAX-functional "
            f"equivalent (e.g., shape/device queries become Python-side metadata, "
            f"`.long()` becomes an explicit `astype(int64)` cast, etc.)."
        )
    if category.startswith("missing_kwarg:"):
        kw = category.split(":", 1)[1]
        return (
            f"Algorithmic fix: change call site to use the positional/alternative form "
            f"that does not pass `{kw}=...`, or (upstream) extend the torch2jax "
            f"interceptor to forward the kwarg."
        )
    if category == "unhashable":
        return (
            "Algorithmic fix: avoid using the traced tensor as a dict key (replace with "
            "an int/string identifier, or rewrite the indexing pattern)."
        )
    return "Algorithmic fix: case-specific; inspect error.txt and reference gold_jax_code.py."


def evaluate_case(tier: str, case_name: str, input_keys, output_keys):
    case_dir = REPO_ROOT / tier / case_name
    inputs = dict(np.load(case_dir / "inputs.npz"))
    expected = dict(np.load(case_dir / "expected.npz"))

    error_text = ""
    status = "PASS"
    max_diffs = {}
    overall_diff = None

    try:
        compute = make_compute(case_dir, output_keys, input_keys)
    except Exception:
        error_text = traceback.format_exc()
        return {
            "case": f"{tier}/{case_name}",
            "tier": tier, "case_name": case_name,
            "status": "ADAPTER_ERROR",
            "max_diff": None, "max_diffs_per_output": {},
            "error_text": error_text,
            "error_category": _categorize(error_text),
        }

    try:
        actual = compute(inputs)
    except Exception:
        error_text = traceback.format_exc()
        return {
            "case": f"{tier}/{case_name}",
            "tier": tier, "case_name": case_name,
            "status": "RUNTIME_ERROR",
            "max_diff": None, "max_diffs_per_output": {},
            "error_text": error_text,
            "error_category": _categorize(error_text),
        }

    for k in expected:
        if k not in actual:
            error_text = f"output key '{k}' missing from compute() result"
            return {
                "case": f"{tier}/{case_name}",
                "tier": tier, "case_name": case_name,
                "status": "MISSING_KEY",
                "max_diff": None, "max_diffs_per_output": {},
                "error_text": error_text, "error_category": "missing_output_key",
            }
        max_diffs[k] = float(np.abs(np.asarray(actual[k]) - expected[k]).max())

    overall_diff = max(max_diffs.values())
    passed = all(
        np.allclose(np.asarray(actual[k]), expected[k], atol=ATOL, rtol=RTOL)
        for k in expected
    )
    return {
        "case": f"{tier}/{case_name}",
        "tier": tier, "case_name": case_name,
        "status": "PASS" if passed else "FAIL",
        "max_diff": overall_diff,
        "max_diffs_per_output": max_diffs,
        "error_text": "" if passed else f"Output diff {overall_diff:.3e} exceeds atol={ATOL}",
        "error_category": "" if passed else "numerical_diff",
    }


def write_case_artifacts(result: dict):
    """Write runs/<tier>__<case>/ with input PT, gold JAX, error, fix proposal."""
    case = result["case_name"]; tier = result["tier"]
    case_dir = REPO_ROOT / tier / case
    out_dir = ROOT / "runs" / f"{tier}__{case}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate_jax").mkdir(parents=True, exist_ok=True)

    shutil.copy(case_dir / "pytorch_code.py", out_dir / "pytorch_code.py")
    shutil.copy(case_dir / "jax_code.py",     out_dir / "gold_jax_code.py")
    (out_dir / "candidate_jax" / "NOTE.md").write_text(CANDIDATE_NOTE)

    err_path = out_dir / "error.txt"
    if result["status"] == "PASS":
        err_path.write_text("PASS\n")
    else:
        err_path.write_text(result["error_text"] or "<no error text>\n")

    (out_dir / "result.json").write_text(json.dumps({
        "case": result["case"],
        "status": result["status"],
        "max_diff": result["max_diff"],
        "max_diffs_per_output": result["max_diffs_per_output"],
        "error_category": result["error_category"],
        "tool": "torch2jax",
        "tool_version": "0.1.0",
    }, indent=2))

    if result["status"] == "PASS":
        proposal_text = (
            f"# `{result['case']}`\n\n"
            f"**Status:** PASS — no fix step needed.\n\n"
            f"`torch2jax` reproduced the PyTorch reference within tolerance "
            f"(max abs diff = {result['max_diff']:.3e})."
        )
    else:
        proposal_text = (
            f"# Proposed fix step for `{result['case']}`\n\n"
            f"**Status:** {result['status']}\n\n"
            f"**Error category:** `{result['error_category']}`\n\n"
            f"**Proposal:** {_propose_fix(result['error_category'], result['case'])}\n\n"
            f"_Note: torch2jax is deterministic, so this proposal is documentation only — "
            f"no fix is automatically applied. The actual fix-step iteration loop "
            f"(captured in `fix_steps/`) only applies to LLM baselines._\n"
        )
    (out_dir / "fix_step_proposal.md").write_text(proposal_text)


def main():
    runs_dir = ROOT / "runs"
    if runs_dir.exists():
        shutil.rmtree(runs_dir)
    runs_dir.mkdir()

    results = []
    for tier, case, inks, outks in CASES:
        print(f"  running {tier}/{case} ...", end=" ", flush=True)
        try:
            r = evaluate_case(tier, case, inks, outks)
        except Exception:
            r = {"case": f"{tier}/{case}", "tier": tier, "case_name": case,
                 "status": "DRIVER_ERROR", "max_diff": None,
                 "max_diffs_per_output": {},
                 "error_text": traceback.format_exc(), "error_category": "driver_error"}
        d = f"{r['max_diff']:.3e}" if r["max_diff"] is not None else "—"
        cat = f" [{r['error_category']}]" if r["error_category"] else ""
        print(f"{r['status']:14s} (diff={d}){cat}")
        write_case_artifacts(r)
        results.append(r)

    # Summary.
    n = len(results)
    passes = sum(1 for r in results if r["status"] == "PASS")
    print(f"\n=== summary ===")
    print(f"  pass@1 = {passes}/{n} = {100 * passes / n:.1f}%")
    cats = {}
    for r in results:
        cats[r["error_category"] or "—"] = cats.get(r["error_category"] or "—", 0) + 1
    for st, c in sorted(cats.items()):
        print(f"  {st}: {c}")

    write_results_md(results, passes, n)
    write_manifest(results, passes, n)


def write_results_md(results, passes, n):
    lines = [
        "# torch2jax baseline results",
        "",
        f"**pass@1 = {passes}/{n} = {100 * passes / n:.1f}%** "
        "(deterministic torch2jax v0.1.0 with one-time format adapter; no LLM iteration).",
        "",
        f"Atol/rtol = 1e-5. See [CONTEXT.md](CONTEXT.md) for tool provenance, "
        "[FINDINGS.md](FINDINGS.md) for paper-facing analysis, and "
        "`runs/<tier>__<case>/` for per-case artifacts (input PT, gold JAX, error log, fix proposal).",
        "",
        "| Case | Status | error_category | max_diff |",
        "|---|---|---|---:|",
    ]
    for r in results:
        d = f"{r['max_diff']:.3e}" if r["max_diff"] is not None else "—"
        cat = r["error_category"] or "—"
        lines.append(f"| `{r['case']}` | {r['status']} | `{cat}` | {d} |")
    (ROOT / "results.md").write_text("\n".join(lines) + "\n")
    print(f"  wrote {ROOT / 'results.md'}")


def write_manifest(results, passes, n):
    manifest = {
        "tool": "torch2jax",
        "tool_version": "0.1.0",
        "atol": ATOL, "rtol": RTOL,
        "pass_at_1": passes / n,
        "n_cases": n, "n_passes": passes,
        "cases": [
            {k: v for k, v in r.items() if k != "error_text"} for r in results
        ],
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  wrote {ROOT / 'manifest.json'}")


if __name__ == "__main__":
    main()
