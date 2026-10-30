"""
Generic n-gram frequency utility, used to build the "filler bank" that the
mimicry-attack constructor (mimicry.py) draws benign padding material from.
Deliberately has no dependency on either the synthetic generator (datagen.py)
or the real-data loader (load_adfa.py) -- it just counts n-grams in whatever
list of token sequences it's given.
"""
from collections import Counter


def common_ngram_bank(sequences, n=3, top_k=60):
    """sequences: list of token sequences (e.g. [e["seq"] for e in entities]).
    Returns the top_k most frequent n-grams, each as a list of tokens."""
    counts = Counter()
    for seq in sequences:
        for i in range(len(seq) - n + 1):
            counts[tuple(seq[i:i + n])] += 1
    return [list(g) for g, _ in counts.most_common(top_k)]
