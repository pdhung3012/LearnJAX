"""CLI entry point.

Examples:

  # Colleague: run cheap tier on one case
  python -m tuandin.t2j_bench.run \
      --tier cheap --case tuandin/jax_translations/e1

  # Colleague: batch over all e-cases
  python -m tuandin.t2j_bench.run \
      --tier cheap --case-glob 'tuandin/jax_translations/e*' \
      --out tuandin/t2j_bench/runs/colleague_cheap.jsonl

  # Principal: premium tier on the hard cases only
  python -m tuandin.t2j_bench.run \
      --tier premium --case-glob 'tuandin/jax_translations/h*' \
      --out tuandin/t2j_bench/runs/principal_premium.jsonl \
      --junior-rounds 4 --senior-rounds 3
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

from .models import TIERS, check_independence
from .openrouter_client import CostLedger, OpenRouterClient
from .pipeline import run_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tier", choices=sorted(TIERS), required=True)
    p.add_argument("--case", type=Path, help="single case dir")
    p.add_argument("--case-glob", type=str, help="glob pattern (relative to repo root)")
    p.add_argument("--out", type=Path,
                   default=Path("tuandin/t2j_bench/runs/trajectories.jsonl"))
    p.add_argument("--work-root", type=Path,
                   default=Path("tuandin/t2j_bench/runs/work"))
    p.add_argument("--junior-rounds", type=int, default=3)
    p.add_argument("--senior-rounds", type=int, default=2)
    return p.parse_args()


def resolve_cases(args) -> list[Path]:
    if args.case and args.case_glob:
        raise SystemExit("use --case OR --case-glob, not both")
    if args.case:
        return [args.case]
    if args.case_glob:
        return [Path(p) for p in sorted(glob.glob(args.case_glob)) if Path(p).is_dir()]
    raise SystemExit("must pass --case or --case-glob")


def main() -> None:
    args = parse_args()
    tier = TIERS[args.tier]

    warns = check_independence(tier)
    for w in warns:
        print(f"[warn] {w}")

    cases = resolve_cases(args)
    if not cases:
        raise SystemExit("no cases matched")

    print(f"[t2j-bench] tier={tier.name} cases={len(cases)} "
          f"translators={len(tier.weak_translators)} → out={args.out}")

    ledger = CostLedger()
    client = OpenRouterClient(ledger=ledger)
    run_batch(
        case_dirs=cases,
        tier=tier,
        client=client,
        work_root=args.work_root,
        out_jsonl=args.out,
        junior_rounds=args.junior_rounds,
        senior_rounds=args.senior_rounds,
    )
    print()
    print("=== cost summary ===")
    print(ledger.summary())


if __name__ == "__main__":
    main()
