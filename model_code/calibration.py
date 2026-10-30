"""
Threshold calibration for both Tier 1 (single score, L / L_low) and Tier 3's
temporal-memory gate (L_mem / L_low_mem), derived from a held-out slice of
real normal traces that is NEVER used for baseline-building or evaluation
(kept as its own three-way split in run_ablation_real.py: train_normal ->
baseline, calib_normal -> here, eval_normal -> the only thing metrics are
ever computed against).

This replaces an earlier version of this pipeline where L/L_LOW were
hardcoded constants (0.64 / 0.50) with a docstring claiming they came from
"a full threshold sweep on a calibration split" -- that sweep did not
actually exist in the code, which both understated how those numbers were
chosen and, in practice, produced a badly miscalibrated gate (Cascade-
Guard's escalation rate came out over 80% on real data, which defeats the
entire point of the cascade -- if almost everything escalates, Stage 0
isn't actually filtering anything). This module fixes that gap directly:
the constants are now the *output* of a real, reproducible procedure, not a
disconnected number sitting next to a docstring that claimed otherwise.
"""
import numpy as np
from shadowkube_tier1 import normalized_distance
from tier3_cascade import stage0_score_real


def calibrate_full_score(calib_normal_seqs, baseline, low_pct=80, high_pct=97.5):
    """Score every calibration-normal trace once; L is the high_pct
    percentile of that score distribution (targets a ~(100-high_pct)% false
    positive rate on held-out normal traffic); L_low is the low_pct
    percentile (defines the bottom of the escalation band -- entities scoring
    below this are confidently benign at Stage 0 and never escalate)."""
    scores = np.array([normalized_distance(seq, baseline) for seq in calib_normal_seqs])
    L = float(np.percentile(scores, high_pct))
    L_low = float(np.percentile(scores, low_pct))
    return L_low, L, scores


def calibrate_memory_score(calib_normal_seqs, baseline, normal_pool, low_pct=80, high_pct=97.5):
    """Same idea, but for Stage 3's EWMA temporal-memory statistic, which has
    a different natural scale than the full-sequence score (segment-level
    scoring against smaller per-segment baselines is noisier) and must
    therefore be calibrated separately rather than derived as a fixed
    multiple of L/L_low."""
    mems = []
    for seq in calib_normal_seqs:
        _, mem = stage0_score_real(seq, baseline, normal_pool, n_chunks=3, alpha=0.5)
        mems.append(mem)
    mems = np.array(mems)
    L_mem = float(np.percentile(mems, high_pct))
    L_low_mem = float(np.percentile(mems, low_pct))
    return L_low_mem, L_mem, mems


def calibrate_all(calib_normal_seqs, baseline, normal_pool, calib_attack_seqs=None,
                   low_pct=70, high_pct=97.5, verbose=True):
    """If calib_attack_seqs is given, L (and L_mem) are chosen by the labeled
    G-Means sweep above -- the correct, honest procedure. Without it, falls
    back to a normal-only percentile heuristic, which the real-data run in
    this package showed gives very poor recall (real ADFA-LD's normal and
    attack score distributions overlap far more than the earlier synthetic
    corpus's did) -- kept only as a documented fallback, not the default
    path used by run_ablation_real.py."""
    L_low, L_pctl, full_scores = calibrate_full_score(calib_normal_seqs, baseline, low_pct, high_pct)
    L_low_mem, L_mem_pctl, mem_scores = calibrate_memory_score(
        calib_normal_seqs, baseline, normal_pool, low_pct, high_pct)

    if calib_attack_seqs is not None:
        L, sweep_table = sweep_for_best_gmeans(calib_normal_seqs, calib_attack_seqs, baseline,
                                                verbose=verbose)
        L_low = float(np.percentile(full_scores, low_pct))
        if L_low >= L:
            L_low = L * 0.85  # keep the escalation band non-empty if the sweep landed low

        attack_mems = []
        for s in calib_attack_seqs:
            _, m = stage0_score_real(s, baseline, normal_pool, n_chunks=3, alpha=0.5)
            attack_mems.append(m)
        attack_mems = np.array(attack_mems)
        g, tpr, fpr = None, None, None
        best = None
        for t in np.linspace(min(mem_scores.min(), attack_mems.min()),
                              max(mem_scores.max(), attack_mems.max()), 60):
            g, tpr, fpr = _gmeans_at(mem_scores, attack_mems, t)
            if best is None or g > best[1]:
                best = (t, g, tpr, fpr)
        L_mem = best[0]
        L_low_mem = float(np.percentile(mem_scores, low_pct))
        if L_low_mem >= L_mem:
            L_low_mem = L_mem * 0.85
        if verbose:
            print(f"    memory sweep: best L_mem={L_mem:.4f}  G-Means={best[1]:.3f}  "
                  f"Recall={best[2]:.3f}  FPR={best[3]:.3f}")
    else:
        L, L_mem = L_pctl, L_mem_pctl
        if verbose:
            print(f"  (percentile-only fallback -- no calib_attack_seqs given)")
            print(f"    full-score:  L_low={L_low:.4f}  L={L:.4f}")
            print(f"    memory:      L_low_mem={L_low_mem:.4f}  L_mem={L_mem:.4f}")

    return {"L_low": L_low, "L": L, "L_low_mem": L_low_mem, "L_mem": L_mem}


def _gmeans_at(scores_normal, scores_attack, thresh):
    tp = (scores_attack > thresh).sum()
    fn = (scores_attack <= thresh).sum()
    fp = (scores_normal > thresh).sum()
    tn = (scores_normal <= thresh).sum()
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return (tpr * tnr) ** 0.5, tpr, 1 - tnr


def sweep_for_best_gmeans(calib_normal_seqs, calib_attack_seqs, baseline,
                           n_candidates=60, verbose=True):
    """A genuine labeled threshold sweep -- unlike calibrate_full_score()
    (which only sees normal data and targets a fixed FPR percentile), this
    uses a small held-out slice of REAL, labeled attack traces (kept
    strictly separate from whatever is later used for evaluation) to find
    the L that actually maximizes G-Means on real data, i.e. what the
    original (unimplemented) docstring claim in run_ablation_real.py should
    have been doing all along. Returns the chosen L plus the full sweep
    table, so the honest recall/FPR trade-off on real data is visible rather
    than hidden behind a single chosen number."""
    normal_scores = np.array([normalized_distance(s, baseline) for s in calib_normal_seqs])
    attack_scores = np.array([normalized_distance(s, baseline) for s in calib_attack_seqs])
    lo = min(normal_scores.min(), attack_scores.min())
    hi = max(normal_scores.max(), attack_scores.max())
    candidates = np.linspace(lo, hi, n_candidates)
    table = []
    best = None
    for t in candidates:
        g, tpr, fpr = _gmeans_at(normal_scores, attack_scores, t)
        table.append((t, g, tpr, fpr))
        if best is None or g > best[1]:
            best = (t, g, tpr, fpr)
    if verbose:
        print(f"  Labeled sweep over {len(candidates)} thresholds "
              f"({len(calib_normal_seqs)} calib-normal + {len(calib_attack_seqs)} calib-attack, "
              f"held out from eval):")
        print(f"    best: L={best[0]:.4f}  G-Means={best[1]:.3f}  "
              f"Recall={best[2]:.3f}  FPR={best[3]:.3f}")
        print(f"    for reference -- normal_scores: mean={normal_scores.mean():.3f}, "
              f"attack_scores: mean={attack_scores.mean():.3f} "
              f"(large overlap here means recall/FPR cannot both be pushed low "
              f"simultaneously; this IS the finding, not a bug)")
    return best[0], table
