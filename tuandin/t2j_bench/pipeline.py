"""Translate → oracle → fix loop with junior/senior escalation.

Each attempt is logged as a bug/fix trajectory record in JSONL, matching
the schema the paper needs:
  {pytorch_code, jax_bug, jax_fix, error, fixer, round}
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from . import prompts
from .models import Model, Tier
from .openrouter_client import OpenRouterClient
from .oracle import OracleResult, run as run_oracle


@dataclass
class TrajectoryStep:
    round: int
    role: str                # "translator" | "junior" | "senior"
    model_slug: str
    jax_code: str
    oracle_status: str
    oracle_error: str | None
    max_diff: float | None
    cost: float
    latency_s: float


@dataclass
class CaseReport:
    case_id: str
    translator_slug: str
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    final_status: str = "UNKNOWN"     # PASS / UNRESOLVED / TRIVIAL
    total_cost: float = 0.0

    def to_json(self) -> str:
        return json.dumps({
            "case_id": self.case_id,
            "translator": self.translator_slug,
            "final_status": self.final_status,
            "total_cost": self.total_cost,
            "trajectory": [asdict(s) for s in self.trajectory],
        })


def _write_and_verify(
    work_dir: Path,
    jax_code: str,
) -> OracleResult:
    (work_dir / "jax_code.py").write_text(jax_code)
    return run_oracle(work_dir)


def _prep_workdir(case_dir: Path, work_root: Path, tag: str) -> Path:
    """Copy the case's inputs.npz + expected.npz into an isolated workdir."""
    work = work_root / f"{case_dir.name}__{tag}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for f in ("inputs.npz", "expected.npz", "pytorch_code.py"):
        src = case_dir / f
        if src.exists():
            shutil.copy2(src, work / f)
    return work


def run_case(
    *,
    case_dir: Path,
    work_root: Path,
    tier: Tier,
    client: OpenRouterClient,
    translator: Model,
    junior_rounds: int = 3,
    senior_rounds: int = 2,
    translator_temp: float = 0.7,
    fixer_temp: float = 0.3,
) -> CaseReport:
    resolved = tier.resolve()
    junior, senior = resolved["junior"], resolved["senior"]

    pytorch_code = (case_dir / "pytorch_code.py").read_text()
    schema = prompts.describe_schema(case_dir)
    work_dir = _prep_workdir(case_dir, work_root, translator.slug.replace("/", "_"))
    report = CaseReport(case_id=case_dir.name, translator_slug=translator.slug)

    # Round 0: translation
    call = client.chat(
        translator,
        [{"role": "system", "content": prompts.SYSTEM_TRANSLATE},
         {"role": "user", "content": prompts.USER_TRANSLATE.format(
             pytorch_code=pytorch_code, schema=schema)}],
        temperature=translator_temp,
    )
    jax_code = prompts.extract_code(call.text)
    oracle = _write_and_verify(work_dir, jax_code)
    report.trajectory.append(TrajectoryStep(
        round=0, role="translator", model_slug=translator.slug, jax_code=jax_code,
        oracle_status=oracle.status, oracle_error=oracle.error, max_diff=oracle.max_diff,
        cost=call.cost, latency_s=call.latency_s,
    ))
    report.total_cost += call.cost

    if oracle.passed:
        report.final_status = "TRIVIAL"   # weak model got it first try — not useful data
        return report

    # Junior fixing rounds
    current_code = jax_code
    for r in range(1, junior_rounds + 1):
        current_code, oracle, call = _fix_once(
            client, junior, pytorch_code, current_code, oracle,
            work_dir, temperature=fixer_temp, schema=schema,
        )
        report.trajectory.append(TrajectoryStep(
            round=r, role="junior", model_slug=junior.slug, jax_code=current_code,
            oracle_status=oracle.status, oracle_error=oracle.error, max_diff=oracle.max_diff,
            cost=call.cost, latency_s=call.latency_s,
        ))
        report.total_cost += call.cost
        if oracle.passed:
            report.final_status = "PASS"
            return report

    # Senior escalation
    for r in range(1, senior_rounds + 1):
        current_code, oracle, call = _fix_once(
            client, senior, pytorch_code, current_code, oracle,
            work_dir, temperature=fixer_temp, schema=schema,
        )
        report.trajectory.append(TrajectoryStep(
            round=junior_rounds + r, role="senior", model_slug=senior.slug, jax_code=current_code,
            oracle_status=oracle.status, oracle_error=oracle.error, max_diff=oracle.max_diff,
            cost=call.cost, latency_s=call.latency_s,
        ))
        report.total_cost += call.cost
        if oracle.passed:
            report.final_status = "PASS"
            return report

    report.final_status = "UNRESOLVED"
    return report


def _fix_once(
    client: OpenRouterClient,
    fixer: Model,
    pytorch_code: str,
    jax_code: str,
    oracle: OracleResult,
    work_dir: Path,
    *,
    temperature: float,
    schema: str,
):
    hint = prompts.diff_hint(oracle.max_diff, oracle.per_output)
    call = client.chat(
        fixer,
        [{"role": "system", "content": prompts.SYSTEM_FIX},
         {"role": "user", "content": prompts.USER_FIX.format(
             pytorch_code=pytorch_code,
             jax_code=jax_code,
             schema=schema,
             status=oracle.status,
             error=(oracle.error or "")[-1500:],
             diff_hint=hint,
         )}],
        temperature=temperature,
    )
    new_code = prompts.extract_code(call.text)
    if not new_code.strip():
        # Fixer produced no code — keep the buggy version and let the loop
        # move on (this round is effectively a no-op but is still logged).
        return jax_code, oracle, call
    new_oracle = _write_and_verify(work_dir, new_code)
    return new_code, new_oracle, call


def run_batch(
    *,
    case_dirs: Iterable[Path],
    tier: Tier,
    client: OpenRouterClient,
    work_root: Path,
    out_jsonl: Path,
    junior_rounds: int = 3,
    senior_rounds: int = 2,
) -> None:
    """Run every (case × weak translator) pair; append trajectories to JSONL."""
    resolved = tier.resolve()
    translators = resolved["weak"]
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("a") as f:
        for case_dir in case_dirs:
            for t in translators:
                report = run_case(
                    case_dir=case_dir, work_root=work_root, tier=tier, client=client,
                    translator=t, junior_rounds=junior_rounds, senior_rounds=senior_rounds,
                )
                f.write(report.to_json() + "\n")
                f.flush()
                print(f"[{report.case_id} × {t.slug}] → {report.final_status} "
                      f"(steps={len(report.trajectory)}, cost=${report.total_cost:.4f})")
