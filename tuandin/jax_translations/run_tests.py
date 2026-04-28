"""Run test_equivalence.py for every case (or a chosen subset).

Usage:
    python run_tests.py                  # run all 30
    python run_tests.py e1 m9 h11        # run subset
    python run_tests.py --random 5       # run 5 random cases
"""
import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = "/opt/miniconda3/envs/t2j/bin/python"
ALL_CASES = sorted(
    [d.name for d in ROOT.iterdir()
     if d.is_dir() and (d / "test_equivalence.py").exists()],
    key=lambda n: (n[0], int(n[1:])),
)


def run_one(case: str, timeout: int = 300):
    path = ROOT / case / "test_equivalence.py"
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    p = subprocess.run([PYTHON, str(path)], capture_output=True, text=True,
                       env=env, cwd=str(path.parent), timeout=timeout)
    return p.returncode == 0, p.stdout, p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", help="cases to run; default: all")
    ap.add_argument("--random", type=int, default=None, help="run N random cases")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.random is not None:
        if args.seed is not None:
            random.seed(args.seed)
        cases = random.sample(ALL_CASES, k=min(args.random, len(ALL_CASES)))
    else:
        cases = args.cases or ALL_CASES

    failed = []
    print(f"running {len(cases)} test_equivalence.py: {cases}")
    print()
    for case in cases:
        if case not in ALL_CASES:
            print(f"  {case:>4}: SKIP (no test file)")
            continue
        ok, out, err = run_one(case)
        status = "PASS" if ok else "FAIL"
        last_line = out.strip().splitlines()[-1] if out.strip() else "<no stdout>"
        print(f"  {case:>4}: {status:5}  {last_line}")
        if not ok:
            failed.append(case)
            print("---- stdout ----")
            print(out[-1500:])
            print("---- stderr ----")
            print(err[-1500:])

    print()
    print(f"{len(cases) - len(failed)}/{len(cases)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
