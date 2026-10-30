"""
Main ablation runner -- REAL ADFA-LD DATA VERSION.

Same three-tier structure and latency-accounting rules as run_ablation.py
(the synthetic version); see that file's docstring for the full explanation
of what's charged to per-event latency vs. treated as offline/periodic.

Key differences from the synthetic version, all driven by real data having
no pod/namespace metadata:
  - ONE global baseline group (load_adfa.GLOBAL_GROUP), not per-scenario.
  - Mimicry construction wraps the *whole* real attack trace in real benign
    padding (mimicry.mimic_attack_real) rather than re-embedding a known
    synthetic signature.
  - Graph features exclude any attack-category encoding (graph_utils.
    build_features_real) to avoid leaking the label into the input.
  - Stage 0's temporal memory uses real-trace-sliced chunk baselines
    (tier3_cascade.stage0_score_real).
  - L / L_low / L_mem / L_low_mem are calibrated by a genuine labeled G-Means
    sweep (calibration.py) on a held-out slice of real normal traces PLUS a
    held-out slice of real attack traces (carved out of attack_entities
    before any mimicry construction, so the calibration attacker traces are
    never the ones later diluted or evaluated on) -- not a normal-only
    percentile heuristic. That heuristic was tried first and gave a
    near-zero-recall operating point on real data (see README): real
    ADFA-LD's normal and attack score distributions overlap far more than
    the earlier synthetic corpus's did, which is itself a real, reportable
    finding about how much harder genuine system data is than a clean
    synthetic benchmark.
"""
import os
import time
import json
import csv
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

from load_adfa import load_adfa, GLOBAL_GROUP
from shadowkube_tier1 import build_baseline_from_entities, normalized_distance, metrics
from mimicry import build_mimicked_entities_real
from graph_utils import (build_features_real, corpus_rare_tokens_real,
                          build_adjacency, normalize_adjacency)
from tier2_gcn import run_tier2
from tier3_cascade import (cutoff_normalized_distance, stage0_score_real, gate,
                            camouflage_weighted_adjacency, precompute_propagation,
                            train_stage1_classifier, stage1_predict)
from ngram_utils import common_ngram_bank
from calibration import calibrate_all

SEED = 42

N_BASELINE = 200
N_CALIB_NORMAL = 400
N_EVAL_NORMAL = 1200   # subsampled from the 4372 real validation normal traces, for tractable runtime
CALIB_ATTACK_FRAC = 0.15  # held out from attack_entities BEFORE any mimicry construction
TAU = 0.90             # cosine-similarity edge threshold (re-checked for real data below)


def pct(arr, p):
    return float(np.percentile(arr, p)) if len(arr) else float("nan")


def run_tier1(entities, labels, baseline, L, eval_idx=None):
    """eval_idx restricts scoring to a specific subset of entities -- used so
    Tier 1/2/3 are all compared on the exact same held-out partition (see
    main(), which computes one shared test split and passes it to all three
    tiers). Tier 1 has no trainable parameters fit on these entities, so
    scoring it on the full set would not itself be leakage, but restricting
    it too makes the three-way comparison apples-to-apples rather than
    comparing different populations."""
    idx_iter = range(len(entities)) if eval_idx is None else eval_idx
    lat, preds, true_sub = [], [], []
    for i in idx_iter:
        e = entities[i]
        t0 = time.perf_counter()
        d = normalized_distance(e["seq"], baseline)
        lat.append((time.perf_counter() - t0) * 1000.0)
        preds.append(1 if d > L else 0)
        true_sub.append(labels[i])
    preds = np.array(preds)
    true_sub = np.array(true_sub)
    tp = int(((preds == 1) & (true_sub == 1)).sum())
    fp = int(((preds == 1) & (true_sub == 0)).sum())
    fn = int(((preds == 0) & (true_sub == 1)).sum())
    tn = int(((preds == 0) & (true_sub == 0)).sum())
    m = metrics(tp, fp, fn, tn)
    m["p50_ms"], m["p95_ms"] = pct(lat, 50), pct(lat, 95)
    return m


def run_tier2_condition(entities, labels, alphabet, rare_tokens, tau, eval_idx):
    full_feats, _ = build_features_real(entities, alphabet, rare_tokens)

    t0 = time.perf_counter()
    A = build_adjacency(full_feats, tau=tau)
    A_norm = normalize_adjacency(A)
    t_graph = time.perf_counter() - t0

    preds, probs, test_idx, model, A_norm2, Xn = run_tier2(
        full_feats, labels, tau=tau, gamma=0.5, hid_dim=16, epochs=250, seed=SEED,
        eval_idx=eval_idx)

    t0 = time.perf_counter()
    _ = model.predict_proba(A_norm2, Xn)
    t_infer_all = time.perf_counter() - t0
    per_node_infer_ms = (t_infer_all / len(labels)) * 1000.0

    y_test = labels[test_idx]
    tp = int(((preds == 1) & (y_test == 1)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    fn = int(((preds == 0) & (y_test == 1)).sum())
    tn = int(((preds == 0) & (y_test == 0)).sum())
    m = metrics(tp, fp, fn, tn)
    per_decision_ms = ((t_graph / len(labels)) * 1000.0 + per_node_infer_ms)
    m["p50_ms"] = m["p95_ms"] = per_decision_ms
    m["graph_construction_s_total"] = t_graph
    return m


def run_tier3_condition(entities, labels, alphabet, rare_tokens, baseline, normal_pool, tau,
                         eval_idx, L_LOW, L, L_low_mem, L_mem):
    full_feats, rare_feats = build_features_real(entities, alphabet, rare_tokens)

    stage0_lat, decisions = [], []
    for e in entities:
        t0 = time.perf_counter()
        full_d, mem = stage0_score_real(e["seq"], baseline, normal_pool, n_chunks=3, alpha=0.5)
        stage0_lat.append((time.perf_counter() - t0) * 1000.0)
        decision, _ = gate(full_d, mem, L_LOW, L, L_low_mem=L_low_mem, L_mem=L_mem)
        decisions.append(decision)

    escalate_idx = np.array([i for i, d in enumerate(decisions) if d == "escalate"])
    escalation_rate = len(escalate_idx) / len(decisions)

    t0 = time.perf_counter()
    A_trust = camouflage_weighted_adjacency(full_feats, rare_feats, tau=tau, rare_tau=0.3)
    P = precompute_propagation(A_trust, full_feats, K=2)
    t_offline = time.perf_counter() - t0

    # IMPORTANT: use the SAME held-out partition as Tier 1/2 (passed in as
    # eval_idx), not an independently-drawn split -- Stage 1's classifier is
    # trained on everything NOT in eval_idx, and final metrics below are
    # computed ONLY over eval_idx, so a node the classifier was trained on
    # never counts toward its own reported accuracy.
    eval_idx_set = set(eval_idx.tolist())
    train_mask = np.array([i not in eval_idx_set for i in range(len(labels))])
    clf, Xn = train_stage1_classifier(P, rare_feats, labels, train_mask, seed=SEED)

    stage1_lat = []
    final_preds = np.zeros(len(labels), dtype=int)
    for i, d in enumerate(decisions):
        if d == "benign":
            final_preds[i] = 0
        elif d == "malicious":
            final_preds[i] = 1
        else:
            t0 = time.perf_counter()
            pred, _ = stage1_predict(clf, Xn, np.array([i]))
            if i in eval_idx_set:
                stage1_lat.append((time.perf_counter() - t0) * 1000.0)
            final_preds[i] = int(pred[0])

    eval_preds = final_preds[eval_idx]
    eval_labels = labels[eval_idx]
    tp = int(((eval_preds == 1) & (eval_labels == 1)).sum())
    fp = int(((eval_preds == 1) & (eval_labels == 0)).sum())
    fn = int(((eval_preds == 0) & (eval_labels == 1)).sum())
    tn = int(((eval_preds == 0) & (eval_labels == 0)).sum())
    m = metrics(tp, fp, fn, tn)

    eval_decisions = [decisions[i] for i in eval_idx]
    eval_stage0_lat = [stage0_lat[i] for i in eval_idx]
    stage1_mean = float(np.mean(stage1_lat)) if stage1_lat else 0.0
    combined_lat = [s0 + (stage1_mean if dec == "escalate" else 0.0)
                    for s0, dec in zip(eval_stage0_lat, eval_decisions)]
    m["p50_ms"], m["p95_ms"] = pct(combined_lat, 50), pct(combined_lat, 95)
    m["escalation_rate"] = sum(1 for d in eval_decisions if d == "escalate") / len(eval_decisions)
    m["stage0_p50_ms"] = pct(eval_stage0_lat, 50)
    m["stage1_mean_ms"] = stage1_mean
    m["offline_setup_s"] = t_offline
    return m


def main():
    print("Loading real ADFA-LD data...")
    train_normal, test_normal, attack_entities = load_adfa()
    print(f"  train_normal={len(train_normal)} test_normal={len(test_normal)} "
          f"attack={len(attack_entities)}")

    r = np.random.default_rng(SEED)
    idx = r.permutation(len(test_normal))
    calib_normal = [test_normal[i] for i in idx[:N_CALIB_NORMAL]]
    remaining = [test_normal[i] for i in idx[N_CALIB_NORMAL:]]
    eval_idx_normal = r.choice(len(remaining), size=min(N_EVAL_NORMAL, len(remaining)), replace=False)
    eval_normal = [remaining[i] for i in eval_idx_normal]

    # Held out BEFORE any mimicry construction: calibration must reflect the
    # detector's real, honest decision boundary against genuine (undiluted)
    # attack behavior, not against traces the attacker has already tampered
    # with. This calibration slice is never reused for evaluation below.
    idx_a = r.permutation(len(attack_entities))
    n_calib_attack = int(CALIB_ATTACK_FRAC * len(attack_entities))
    calib_attack_entities = [attack_entities[i] for i in idx_a[:n_calib_attack]]
    eval_attack_entities = [attack_entities[i] for i in idx_a[n_calib_attack:]]
    print(f"  attack split: {len(calib_attack_entities)} for threshold calibration, "
          f"{len(eval_attack_entities)} held out for evaluation")

    baseline = build_baseline_from_entities(train_normal, max_exemplars=N_BASELINE, seed=0)
    normal_pool_seqs = [e["seq"] for e in train_normal]

    all_seen = train_normal + test_normal + attack_entities
    alphabet = sorted({tok for e in all_seen for tok in e["seq"]}, key=lambda x: int(x))
    rare_tokens = corpus_rare_tokens_real(train_normal + attack_entities, alphabet, k=15)
    print(f"  alphabet size={len(alphabet)}, rare tokens={rare_tokens}")

    filler_bank = common_ngram_bank([e["seq"] for e in train_normal], n=3, top_k=60)

    print("Calibrating detection thresholds (labeled G-Means sweep)...")
    calib_normal_seqs = [e["seq"] for e in calib_normal]
    calib_attack_seqs = [e["seq"] for e in calib_attack_entities]
    params = calibrate_all(calib_normal_seqs, baseline, normal_pool_seqs,
                            calib_attack_seqs=calib_attack_seqs)
    L, L_LOW = params["L"], params["L_low"]
    L_mem, L_low_mem = params["L_mem"], params["L_low_mem"]

    for e in eval_attack_entities:
        e["D"] = normalized_distance(e["seq"], baseline)
    for e in eval_normal:
        e["D"] = normalized_distance(e["seq"], baseline)

    rows = []
    for mode in ["original", "mimicked"]:
        if mode == "original":
            atk = eval_attack_entities
        else:
            print("  constructing real-data mimicry set (this takes ~1-3 min)...")
            atk = build_mimicked_entities_real(
                eval_attack_entities, baseline, filler_bank, normal_pool_seqs, L=L,
                max_inflation=5.0, length_steps=10, hillclimb_iters=50, seed=SEED)
            for e in atk:
                e["D"] = e["mimicry_score"]
            n_evaded = sum(1 for e in atk if e.get("mimicry_evaded_tier1"))
            print(f"  mimicry evaded Tier1 for {n_evaded}/{len(atk)} attack traces "
                  f"({100 * n_evaded / len(atk):.1f}%)")

        entities = atk + eval_normal
        labels = np.array([1 if e["true"] == "attack" else 0 for e in entities])
        print(f"\n=== Condition: {mode} (n={len(entities)}, "
              f"n_attack={int(labels.sum())}, n_normal={int((labels == 0).sum())}) ===")

        # ONE shared held-out partition, used identically by all three tiers
        # (fixed by position, so it's the same logical subset of entities in
        # both the "original" and "mimicked" conditions too, since entity
        # order is preserved between conditions).
        r_split = np.random.default_rng(SEED)
        perm = r_split.permutation(len(labels))
        eval_idx = np.sort(perm[:int(0.3 * len(labels))])

        m1 = run_tier1(entities, labels, baseline, L, eval_idx=eval_idx)
        print("Tier1:", {k: round(v, 3) if isinstance(v, float) else v for k, v in m1.items()})
        rows.append({"condition": mode, "tier": "Tier1_ShadowKube", **m1})

        m2 = run_tier2_condition(entities, labels, alphabet, rare_tokens, TAU, eval_idx)
        print("Tier2:", {k: round(v, 3) if isinstance(v, float) else v for k, v in m2.items()})
        rows.append({"condition": mode, "tier": "Tier2_GuardNet_naive", **m2})

        m3 = run_tier3_condition(entities, labels, alphabet, rare_tokens, baseline,
                                  normal_pool_seqs, TAU, eval_idx, L_LOW, L, L_low_mem, L_mem)
        print("Tier3:", {k: round(v, 3) if isinstance(v, float) else v for k, v in m3.items()})
        rows.append({"condition": mode, "tier": "Tier3_CascadeGuard", **m3})

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(os.path.join(OUTPUT_DIR, "ablation_results_real.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(OUTPUT_DIR, "ablation_results_real.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'ablation_results_real.csv')} / "
          f"{os.path.join(OUTPUT_DIR, 'ablation_results_real.json')}")
    return rows


if __name__ == "__main__":
    main()