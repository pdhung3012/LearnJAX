"""Byte-Pair Encoding (BPE) tokenizer — pure-Python algorithm used by
GPT-2/GPT-3/Llama tokenizers under the hood.

Source: TorchLeet llm/Byte-Pair-Encoder/BPE-q3-SOLN.ipynb.

Note: this problem doesn't actually use PyTorch — it's pure Python. The file
is named pytorch_code.py only to keep the directory layout uniform.
"""
from collections import defaultdict, Counter


def get_vocab(corpus):
    """Build a frequency vocabulary with words split into characters
    and a special end-of-word token."""
    vocab = Counter()
    for word in corpus:
        tokens = list(word) + ["</w>"]
        vocab[tuple(tokens)] += 1
    return vocab


def get_stats(vocab):
    """Count frequencies of adjacent symbol pairs."""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_vocab(pair, vocab):
    """Merge the most frequent pair into a single symbol."""
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word, freq in vocab.items():
        word_str = " ".join(word)
        new_word_str = word_str.replace(bigram, replacement)
        new_vocab[tuple(new_word_str.split())] = freq
    return new_vocab


def byte_pair_encoding(corpus, num_merges=10):
    vocab = get_vocab(corpus)
    merges = []
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        merges.append(best)
        print(f"Merge {i + 1}: {best}")
    return vocab, merges


if __name__ == "__main__":
    corpus = ["low", "lowest", "newer", "wider"]
    final_vocab, merge_operations = byte_pair_encoding(corpus, num_merges=10)
    print("\nFinal Vocabulary:")
    for word in final_vocab:
        print(" ".join(word), ":", final_vocab[word])

    # Tests
    assert get_vocab(["test"]) == {("t", "e", "s", "t", "</w>"): 1}
    assert get_stats({("t", "e", "s", "t", "</w>"): 1}) == {
        ("t", "e"): 1, ("e", "s"): 1, ("s", "t"): 1, ("t", "</w>"): 1,
    }
    assert merge_vocab(("e", "s"), {("t", "e", "s", "t", "</w>"): 1}) == {
        ("t", "es", "t", "</w>"): 1
    }
    print("✓ all tests passed")
