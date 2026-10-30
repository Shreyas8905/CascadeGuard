"""
Tier 1 -- ShadowKube's own detector (their Algorithm 2 / Eq. 4), scored
against real ADFA-LD traces (Creech & Hu, 2013).

Simplification note (documented, not hidden): ShadowKube's Algorithm 1
merges per-pod exemplars into a compact feature set via LCS + a min-length
threshold `ms` (Eq. 1-3). We implement the faithful *scoring* side (Eq. 4:
normalized Levenshtein distance to the nearest baseline exemplar) exactly,
and represent the baseline w_a^p as a capped sample of real training
exemplars rather than LCS-compacted patterns -- i.e. a nearest-exemplar
variant of the same scoring rule. This preserves the paper's core detection
logic (distance to nearest known-normal pattern) while keeping the
reference implementation small enough to time precisely.
"""
import numpy as np
from rapidfuzz.distance import Levenshtein as _RFLevenshtein


def levenshtein(a, b):
    """Edit distance -- delegated to rapidfuzz's compiled implementation for
    speed. Algorithmically identical to ShadowKube's stated O(n*m) DP; this
    is purely an implementation-speed choice so the ablation's *own* tooling
    overhead doesn't dominate the latency numbers we're trying to measure."""
    return _RFLevenshtein.distance(a, b)


def normalized_distance(seq, baseline_set):
    """Eq. 4: d = min( lev(e_a^b, s) / |s| for s in w_a^p )"""
    best = 1.0
    for s in baseline_set:
        d = levenshtein(seq, s) / max(1, len(s))
        if d < best:
            best = d
    return best


def build_baseline_from_entities(entities, max_exemplars=200, seed=0):
    """Baseline w_a^p: a capped random sample of real Training_Data_Master
    traces. ADFA-LD's normal training traces already span 79-2948 tokens
    naturally, so a single pool built this way provides the length
    diversity a length-dilution mimicry attack needs to be tested against,
    with no synthetic generation required."""
    r = np.random.default_rng(seed)
    pool = [e["seq"] for e in entities]
    if len(pool) > max_exemplars:
        idx = r.choice(len(pool), size=max_exemplars, replace=False)
        pool = [pool[i] for i in idx]
    return pool


def metrics(tp, fp, fn, tn):
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    tpr = recall
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = 1 - fpr
    gmeans = (tpr * tnr) ** 0.5
    return {"recall": recall, "precision": precision, "f1": f1, "gmeans": gmeans,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}
