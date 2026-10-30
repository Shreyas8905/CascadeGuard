"""
Graph construction over real ADFA-LD trace entities, following GuardNet's
own recipe (Sect. 4.1): cosine similarity over a combined feature vector,
thresholded at tau to form edges. Two feature views per node:

- `full_feat`  : token-frequency histogram over the real observed syscall-
                 number alphabet + length + D-score. This is the view a
                 length-dilution mimicry attack directly corrupts -- padding
                 with real benign filler makes this vector converge toward
                 the benign population's centroid (that's the whole point of
                 dilution).
- `rare_feat`  : token-frequency histogram restricted to the alphabet's
                 rarest syscall numbers corpus-wide -- the closest available
                 analog to "raw sequence signal the attacker isn't directly
                 optimizing against" (Architectural Improvement 4 in the
                 study report).

NOTE: attack category (Adduser/Hydra_SSH/etc.) is deliberately NOT included
as a feature -- every attack-category value only ever co-occurs with
label=attack in this dataset, so one-hot-encoding it would hand the
classifier the label directly rather than testing whether it can detect
attacks from behavior.
"""
import numpy as np
from collections import Counter


def token_histogram(seq, tokens):
    idx = {t: i for i, t in enumerate(tokens)}
    hist = np.zeros(len(tokens))
    for tok in seq:
        j = idx.get(tok)
        if j is not None:
            hist[j] += 1
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist


def corpus_rare_tokens_real(entities, alphabet, k=15):
    """Ranks the actual observed syscall-number alphabet by corpus
    frequency, restricted to the rarest k -- used as the 'hard to dilute'
    auxiliary feature channel."""
    counts = Counter()
    for e in entities:
        counts.update(e["seq"])
    ranked = sorted(alphabet, key=lambda t: counts.get(t, 0))
    return ranked[:k]


def build_features_real(entities, alphabet, rare_tokens):
    full_feats, rare_feats = [], []
    for e in entities:
        hist = token_histogram(e["seq"], tokens=alphabet)
        length_norm = min(len(e["seq"]) / 1000.0, 1.0)
        d_score = e.get("D", e.get("mimicry_score", 0.0))
        full = np.concatenate([hist, [length_norm, d_score]])
        full_feats.append(full)
        rare_feats.append(token_histogram(e["seq"], tokens=rare_tokens))
    return np.vstack(full_feats), np.vstack(rare_feats)


def cosine_sim_matrix(X):
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1e-9
    Xn = X / norm
    return Xn @ Xn.T


def build_adjacency(full_feats, tau=0.90):
    """GuardNet Eq. 1: A_ij = 1 if cosine(v_i, v_j) > tau, else 0."""
    S = cosine_sim_matrix(full_feats)
    A = (S > tau).astype(float)
    np.fill_diagonal(A, 1.0)  # self-loops, standard GCN practice
    return A


def normalize_adjacency(A):
    """D^-1/2 A D^-1/2, GuardNet Eq. 8 / standard GCN normalization."""
    deg = A.sum(axis=1)
    deg[deg == 0] = 1e-9
    d_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    return d_inv_sqrt @ A @ d_inv_sqrt
