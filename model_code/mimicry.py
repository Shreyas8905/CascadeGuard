"""
Mimicry-attack construction on real ADFA-LD traces (Wagner & Soto, CCS 2002):
given a real, labeled attack trace that Tier 1 currently flags, dilute its
normalized edit-distance score by embedding the *entire* real trace inside a
longer stretch of real, resampled normal material, then refine with generic
benign n-gram insertion. Unlike the earlier synthetic-data version, there is
no separate "signature" to extract -- ADFA-LD labels whole traces as attack,
not a short injected marker within an otherwise-normal trace, so the whole
trace is what gets diluted.
"""
import copy
import numpy as np
from shadowkube_tier1 import normalized_distance


def sample_real_substrate(normal_pool, target_len, seed=None):
    """Concatenate real normal traces (with replacement) until reaching
    target_len, then truncate -- a "keep the compromised process busy with
    mundane real work" substrate, built entirely from genuine normal data,
    no synthetic generation."""
    r = np.random.default_rng(seed)
    out = []
    while len(out) < target_len:
        trace = normal_pool[int(r.integers(0, len(normal_pool)))]
        out.extend(trace)
    return out[:target_len]


def mimic_attack(seq, baseline_set, filler_bank, normal_pool, L,
                  max_inflation=3.0, length_steps=8, hillclimb_iters=40,
                  seed=None):
    """
    Realistic mimicry strategy: dilute the fixed edit-cost of the real
    attack trace by embedding it inside a much longer run of real
    legitimate-looking activity, so the normalized distance -- lev(e, s) /
    |s| -- is scored against a longer baseline exemplar `s`, shrinking the
    ratio even though the attack trace's own content hasn't changed at all.
    Tries several inflation factors, then hill-climbs with real n-gram
    filler as a secondary refinement. Returns (mimicked_seq, achieved_score,
    n_insertions, evaded: bool).
    """
    r = np.random.default_rng(seed)
    orig_len = len(seq)

    best_seq, best_score = list(seq), normalized_distance(seq, baseline_set)
    total_insertions = 0

    for step in range(1, length_steps + 1):
        if best_score < L:
            break
        target_len = int(orig_len * (1 + (max_inflation - 1) * step / length_steps))
        substrate = sample_real_substrate(normal_pool, target_len,
                                           seed=int(r.integers(0, 1 << 30)))
        insert_at = int(r.integers(0, max(1, len(substrate) - 1)))
        candidate = substrate[:insert_at] + list(seq) + substrate[insert_at:]
        cand_score = normalized_distance(candidate, baseline_set)
        if cand_score < best_score:
            best_seq, best_score = candidate, cand_score
            total_insertions = len(candidate) - len(seq)

    cur, cur_score = best_seq, best_score
    for _ in range(hillclimb_iters):
        if cur_score < L or len(cur) >= orig_len * max_inflation:
            break
        filler = filler_bank[int(r.integers(0, len(filler_bank)))]
        pos = int(r.integers(0, len(cur) + 1))
        candidate = cur[:pos] + list(filler) + cur[pos:]
        cand_score = normalized_distance(candidate, baseline_set)
        if cand_score <= cur_score:
            cur, cur_score = candidate, cand_score
            total_insertions += 1

    evaded = cur_score < L
    return cur, cur_score, total_insertions, evaded


def build_mimicked_entities(attack_entities, baseline_set, filler_bank, normal_pool,
                             L, max_inflation=3.0, length_steps=8,
                             hillclimb_iters=40, seed=0):
    """Apply mimic_attack() to every ground-truth-attack entity."""
    r = np.random.default_rng(seed)
    out = []
    for e in attack_entities:
        if e["true"] != "attack":
            out.append(copy.deepcopy(e))
            continue
        mseq, score, n_ins, evaded = mimic_attack(
            e["seq"], baseline_set, filler_bank, normal_pool, L,
            max_inflation=max_inflation, length_steps=length_steps,
            hillclimb_iters=hillclimb_iters, seed=int(r.integers(0, 1 << 30)))
        new_e = copy.deepcopy(e)
        new_e["seq"] = mseq
        new_e["mimicry_score"] = score
        new_e["mimicry_insertions"] = n_ins
        new_e["mimicry_evaded_tier1"] = evaded
        out.append(new_e)
    return out


# Alias for the real-data pipeline (run_ablation_real.py) -- identical
# function, real vs. synthetic data both flow through the same logic here
# since this version was already rewritten to use real substrate sampling.
build_mimicked_entities_real = build_mimicked_entities
