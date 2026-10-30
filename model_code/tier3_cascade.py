"""
Tier 3 -- Cascade-Guard, scored against real ADFA-LD traces. Implements (a
first, honest pass at) the five architectural improvements from the study
report:

  1. Accelerated Stage 0 scorer: uses rapidfuzz's `score_cutoff` early-exit
     (the practical equivalent of banded edit-distance -- once the running
     distance provably exceeds the cutoff, computation stops) so the common
     case (clearly benign or clearly malicious) is scored faster than a full
     from-scratch DP pass. This is the real, measurable substitute for full
     Ukkonen banding within this project's time budget.
  2. Confidence-band gating: L_low / L instead of a single threshold.
  3. Latency-bounded Stage 1: SGC-style precomputed propagation (A_norm^K X
     computed once per graph topology) + a cheap logistic-regression head,
     evaluated ONLY on escalated nodes.
  4. Camouflage-resistant aggregation: edge weights discounted by rare-token
     ("hard to dilute") profile dissimilarity before propagation, and the
     rare-token profile is also concatenated into Stage 1's classifier input
     as an auxiliary channel.
  5. Cheap temporal memory: EWMA of fixed-length-segment suspicion scores
     (each segment scored against same-length real material -- comparing a
     short segment to a long baseline exemplar would invalidate the score,
     since edit distance is bounded below by |len(a)-len(b)|), fed into the
     gate alongside the final Stage-0 score. Implemented and measured here;
     the dedicated *paced* mimicry stress test that most directly exercises
     this component is flagged as follow-up work in the report, not claimed
     as validated by this pass.
"""
import numpy as np
from rapidfuzz.distance import Levenshtein as _RFLevenshtein
from graph_utils import cosine_sim_matrix, normalize_adjacency
from sklearn.linear_model import LogisticRegression


# ---------- Stage 0 ----------

def cutoff_normalized_distance(seq, baseline_set, cutoff=0.30):
    """Same scoring rule as Tier 1 (Eq. 4), but each per-exemplar comparison
    is given an absolute score_cutoff derived from the *normalized* cutoff,
    letting rapidfuzz's C implementation early-exit once a candidate clearly
    can't beat the current best -- this is the accelerated Stage-0 path."""
    best = 1.0
    for s in baseline_set:
        abs_cutoff = int(cutoff * len(s)) + 1
        d_abs = _RFLevenshtein.distance(seq, s, score_cutoff=abs_cutoff)
        d = d_abs / max(1, len(s))
        if d < best:
            best = d
    return best


def ewma_memory(chunk_scores, alpha=0.5):
    m = chunk_scores[0]
    for d in chunk_scores[1:]:
        m = alpha * d + (1 - alpha) * m
    return m


def gate(full_d, mem, L_low, L, L_low_mem, L_mem):
    """Escalate if EITHER the instantaneous score or the temporal memory
    falls in its own ambiguous band -- memory catches a rising trend the
    final score alone might not. The memory statistic is calibrated on its
    own scale (L_low_mem/L_mem), separately from the full-sequence score's
    scale (L_low/L), since segment-level scoring against smaller per-segment
    baselines is naturally noisier/higher-variance than full-sequence
    scoring against a larger baseline pool. All four thresholds are expected
    to come from calibration.calibrate_all() -- see run_ablation_real.py."""
    if full_d > L or mem > L_mem:
        return "malicious", max(full_d, mem)
    if full_d < L_low and mem < L_low_mem:
        return "benign", max(full_d, mem)
    return "escalate", max(full_d, mem)


_CHUNK_BASELINE_CACHE_REAL = {}


def _chunk_baseline_real(chunk_len, normal_pool, n_exemplars=10, seed=11):
    """Real-data variant of _chunk_baseline(): instead of a generative motif
    model, slice fixed-length windows out of real normal traces (traces
    shorter than chunk_len are looped, matching _real_substrate's approach
    in mimicry.py) -- same rationale: comparing a short segment against a
    long/short-mismatched baseline invalidates the score via the
    |len(a)-len(b)| lower bound on edit distance, so segments must be
    compared against same-length real material."""
    key = chunk_len
    if key not in _CHUNK_BASELINE_CACHE_REAL:
        r = np.random.default_rng(seed)
        exemplars = []
        for _ in range(n_exemplars):
            trace = normal_pool[int(r.integers(0, len(normal_pool)))]
            if len(trace) >= chunk_len:
                start = int(r.integers(0, len(trace) - chunk_len + 1))
                exemplars.append(trace[start:start + chunk_len])
            else:
                looped = (trace * (chunk_len // len(trace) + 1))[:chunk_len]
                exemplars.append(looped)
        _CHUNK_BASELINE_CACHE_REAL[key] = exemplars
    return _CHUNK_BASELINE_CACHE_REAL[key]


def stage0_score_real(seq, baseline_set, normal_pool, n_chunks=3, alpha=0.5):
    """Real-data variant of stage0_score() -- same EWMA-over-equal-segments
    logic, but segment baselines are built from real normal traces."""
    full_d = cutoff_normalized_distance(seq, baseline_set, cutoff=0.5)
    n = len(seq)
    seg_len = max(5, n // n_chunks)
    chunk_scores = []
    for i in range(0, n - seg_len + 1, seg_len):
        segment = seq[i:i + seg_len]
        cbaseline = _chunk_baseline_real(len(segment), normal_pool)
        chunk_scores.append(cutoff_normalized_distance(segment, cbaseline, cutoff=0.6))
    mem = ewma_memory(chunk_scores, alpha=alpha) if chunk_scores else full_d
    return full_d, mem


# ---------- Stage 1 ----------

def camouflage_weighted_adjacency(full_feats, rare_feats, tau=0.97, rare_tau=0.3):
    """Standard cosine-threshold adjacency on full_feats (defines *which*
    edges exist, same as Tier 2/GuardNet), but each surviving edge's weight
    is discounted by how dissimilar the endpoints' rare-token profiles are.
    A node whose full-feature vector matches its neighbors (diluted by
    filler) but whose rare-token profile diverges gets down-weighted in
    propagation -- this is the CARE-GNN-style "don't trust superficial
    similarity" mechanism (Architectural Improvement 4)."""
    S_full = cosine_sim_matrix(full_feats)
    A = (S_full > tau).astype(float)
    np.fill_diagonal(A, 1.0)
    S_rare = cosine_sim_matrix(rare_feats)
    trust = np.clip(S_rare, 0, 1)
    trust = np.where(trust > rare_tau, trust, 0.15)  # heavy discount, not zero
    A_trust = A * trust
    np.fill_diagonal(A_trust, 1.0)
    return A_trust


def precompute_propagation(A_trust, full_feats, K=2):
    """SGC (Wu et al. 2019): precompute A_norm^K @ X once per graph topology
    -- this is the offline cost; inference for any single node afterward is
    just reading a row of an already-computed matrix."""
    A_norm = normalize_adjacency(A_trust)
    P = full_feats.copy()
    for _ in range(K):
        P = A_norm @ P
    return P


def train_stage1_classifier(P, rare_feats, labels, train_mask, seed=0):
    X = np.concatenate([P, rare_feats], axis=1)
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-6)
    clf = LogisticRegression(max_iter=500, class_weight="balanced",
                              random_state=seed)
    clf.fit(Xn[train_mask], labels[train_mask])
    return clf, Xn


def stage1_predict(clf, Xn, node_indices):
    """The only part of Stage 1 actually charged to per-event latency in the
    ablation: a row lookup + one cheap linear-model inference per escalated
    node."""
    return clf.predict(Xn[node_indices]), clf.predict_proba(Xn[node_indices])[:, 1]
