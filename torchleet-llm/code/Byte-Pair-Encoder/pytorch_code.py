from collections import defaultdict, Counter

def get_vocab(corpus):
    """Creates a vocabulary with words split into characters and a special end-of-word token."""
    vocab = Counter()
    for word in corpus:
        tokens = list(word) + ['']
        vocab[tuple(tokens)] += 1
    return vocab

def get_stats(vocab):
    """Counts frequency of adjacent symbol pairs."""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """Merges the most frequent pair into a single symbol."""
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word, freq in vocab.items():
        word_str = ' '.join(word)
        # Replace bigram with merged symbol
        new_word_str = word_str.replace(bigram, replacement)
        new_vocab[tuple(new_word_str.split())] = freq
    return new_vocab

def byte_pair_encoding(corpus, num_merges=10):
    """Performs BPE on a corpus."""
    vocab = get_vocab(corpus)
    merges = []
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        merges.append(best)
        print(f"Merge {_ + 1}: {best}")
    return vocab, merges

# Example usage
corpus = ["low", "lowest", "newer", "wider"]
final_vocab, merge_operations = byte_pair_encoding(corpus, num_merges=10)

print("\nFinal Vocabulary:")
for word in final_vocab:
    print(' '.join(word), ":", final_vocab[word])

def test_get_vocab():
    corpus = ["test"]
    vocab = get_vocab(corpus)
    assert vocab == {('t', 'e', 's', 't', ''): 1}
    print("✓ test_get_vocab passed")

def test_get_stats():
    vocab = {('t', 'e', 's', 't', ''): 1}
    stats = get_stats(vocab)
    expected = {
        ('t', 'e'): 1,
        ('e', 's'): 1,
        ('s', 't'): 1,
        ('t', ''): 1
    }
    assert stats == expected
    print("✓ test_get_stats passed")

def test_merge_vocab():
    vocab = {('t', 'e', 's', 't', ''): 1}
    merged = merge_vocab(('e', 's'), vocab)
    expected = {('t', 'es', 't', ''): 1}
    assert merged == expected
    print("✓ test_merge_vocab passed")

def test_bpe_sequence():
    corpus = ["low", "lower", "newest", "widest"]
    final_vocab, merges = byte_pair_encoding(corpus, num_merges=5)
    assert isinstance(final_vocab, dict)
    assert all(isinstance(pair, tuple) for pair in merges)
    assert len(merges) == 5
    print("✓ test_bpe_sequence passed")

# Run all tests
test_get_vocab()
