"""Freeze fixtures for e11: BPE produces deterministic merges given a corpus.

BPE has no tensor I/O so we deviate from the .npz convention: inputs.npz holds
the corpus as an object array, and expected.npz stores the merge sequence and
final vocab encoded as numpy object arrays.
"""
import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pytorch_code as ref


def make_inputs():
    corpus = ["lower", "lowest", "newer", "newest", "wider", "widest"]
    return {
        "corpus":     np.array(corpus, dtype=object),
        "num_merges": np.array(8, dtype=np.int32),
    }


def pytorch_reference(inputs):
    corpus = list(inputs["corpus"])
    final_vocab, merges = ref.byte_pair_encoding(corpus, num_merges=int(inputs["num_merges"]))
    # Encode the merge sequence and vocab as JSON-stringified arrays so they
    # round-trip through npz cleanly.
    return {
        "merges_json": np.array(json.dumps([list(m) for m in merges])),
        "vocab_json":  np.array(json.dumps({" ".join(k): v for k, v in final_vocab.items()})),
    }


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("e11: fixtures written")
