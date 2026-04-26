"""e11 equivalence test: BPE produces identical merges/vocabulary on both sides.

The PyTorch and JAX files are byte-identical (BPE is pure Python). We still
exercise the import and assert merge sequences match for a fresh corpus.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pytorch_code as pt_mod
import jax_code as jax_mod


def main():
    corpus = ["lower", "lowest", "newer", "newest", "wider", "widest"]
    vocab_pt, merges_pt = pt_mod.byte_pair_encoding(corpus, num_merges=8)
    vocab_jx, merges_jx = jax_mod.byte_pair_encoding(corpus, num_merges=8)
    assert merges_pt == merges_jx, f"merges differ:\n  pt:  {merges_pt}\n  jax: {merges_jx}"
    assert vocab_pt == vocab_jx, f"vocab differs:\n  pt:  {vocab_pt}\n  jax: {vocab_jx}"
    print("[e11] PASS")


if __name__ == "__main__":
    main()
