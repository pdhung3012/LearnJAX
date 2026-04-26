"""Run pytorch_code.py and jax_code.py for each case, time wall-clock, report.

Skips cases that need to download large datasets (CIFAR10/MNIST) — they are
data-loader-bound so the comparison is uninformative anyway. Also skips the
matplotlib-heavy cases (e3, m1, m6, h10) when run headless.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = "/opt/miniconda3/envs/t2j/bin/python"

# Cases included by default (skip large-download / GUI / GPU-only cases).
# Excluded: m3/m4/m6/m7/m8 (CIFAR/MNIST downloads), m11 (HF model download),
# h13 (Triton needs CUDA), h10 (Grad-CAM with pretrained weights download).
CASES = [
    "e1", "e2", "e3", "e4", "e5", "e6", "e7",
    "e8", "e9", "e10", "e11",
    "m1", "m5", "m9", "m10",
    "h1", "h3", "h4", "h5", "h6",
    "h11", "h12",
]

def run_and_time(path, label):
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"  # never pop a GUI window.
    t0 = time.time()
    proc = subprocess.run(
        [PY, path],
        capture_output=True, text=True, env=env, cwd=os.path.dirname(path),
    )
    dt = time.time() - t0
    ok = proc.returncode == 0
    return dt, ok, proc.stdout[-2000:], proc.stderr[-2000:]

def main():
    cases = sys.argv[1:] or CASES
    print(f"{'case':>6}  {'pytorch_s':>10}  {'jax_s':>10}  {'speedup_jax':>11}")
    for case in cases:
        torch_path = os.path.join(ROOT, case, "pytorch_code.py")
        jax_path = os.path.join(ROOT, case, "jax_code.py")
        if not (os.path.exists(torch_path) and os.path.exists(jax_path)):
            print(f"{case:>6}  missing files")
            continue
        dt_t, ok_t, _, errt = run_and_time(torch_path, f"{case}/pt")
        dt_j, ok_j, _, errj = run_and_time(jax_path, f"{case}/jax")
        if not ok_t:
            print(f"{case:>6}  PYTORCH FAILED")
            print(errt[-400:])
            continue
        if not ok_j:
            print(f"{case:>6}  JAX FAILED")
            print(errj[-400:])
            continue
        speedup = dt_t / dt_j
        print(f"{case:>6}  {dt_t:10.3f}  {dt_j:10.3f}  {speedup:10.2f}x")

if __name__ == "__main__":
    main()
