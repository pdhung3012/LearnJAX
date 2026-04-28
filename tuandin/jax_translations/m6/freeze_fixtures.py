"""m6 is a CIFAR-10 augmentation visualization — there is nothing numeric to
freeze. We still emit empty fixtures so the same `test_equivalence.py`
template applies; `compute()` returns an empty dict.
"""
import numpy as np


if __name__ == "__main__":
    np.savez("inputs.npz")    # empty
    np.savez("expected.npz")  # empty
    print("m6: empty fixtures written (visualization-only case, nothing to test)")
