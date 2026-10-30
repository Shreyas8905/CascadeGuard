"""Train the reusable Tier-3 research bundle from ADFA-LD and export it.

This is a transductive-reference implementation: the saved reference graph is
used when scoring new traces. It is suitable for reproducible experiments and
integration tests, not yet a claim of production throughput or Kubernetes
accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn

from calibration import calibrate_all
from graph_utils import build_features_real, corpus_rare_tokens_real
from load_adfa import load_adfa
from shadowkube_tier1 import build_baseline_from_entities, normalized_distance
from tier3_cascade import (
    camouflage_weighted_adjacency,
    precompute_propagation,
    train_stage1_classifier,
)

SEED = 42
TAU = 0.90
K_PROPAGATION = 2
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "cascadeguard_tier3.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export CascadeGuard Tier 3")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output .joblib bundle path")
    parser.add_argument("--calibration-normal", type=int, default=400)
    parser.add_argument("--calibration-attack-fraction", type=float, default=0.15)
    parser.add_argument("--test-attack-fraction", type=float, default=0.15)
    parser.add_argument("--baseline-exemplars", type=int, default=200)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    print("[1/6] Loading ADFA-LD traces...")
    train_normal, validation_normal, attacks = load_adfa()
    if not train_normal or not validation_normal or not attacks:
        raise SystemExit(
            "ADFA-LD data not found or incomplete. Expected adfa_ld/extracted/ADFA-LD/"
            "Training_Data_Master, Validation_Data_Master, and Attack_Data_Master."
        )
    if not 0.0 < args.calibration_attack_fraction < 0.5:
        raise SystemExit("--calibration-attack-fraction must be between 0 and 0.5")
    if not 0.0 < args.test_attack_fraction < 0.5 or args.calibration_attack_fraction + args.test_attack_fraction >= 0.8:
        raise SystemExit("--test-attack-fraction must be >0 and <0.5; calibration + test fractions must be <0.8")
    if args.calibration_normal < 10 or args.calibration_normal >= len(validation_normal):
        raise SystemExit("--calibration-normal must be >=10 and smaller than validation normal count")

    # Keep calibration traces out of the exported Stage-1 training graph.
    attack_order = rng.permutation(len(attacks))
    n_cal_attack = max(1, int(len(attacks) * args.calibration_attack_fraction))
    n_test_attack = max(1, int(len(attacks) * args.test_attack_fraction))
    calibration_attacks = [attacks[i] for i in attack_order[:n_cal_attack]]
    test_attacks = [attacks[i] for i in attack_order[n_cal_attack:n_cal_attack + n_test_attack]]
    training_attacks = [attacks[i] for i in attack_order[n_cal_attack + n_test_attack:]]
    normal_order = rng.permutation(len(validation_normal))
    calibration_normals = [validation_normal[i] for i in normal_order[:args.calibration_normal]]

    print(f"  train normal={len(train_normal)}, validation normal={len(validation_normal)}, "
          f"attack train={len(training_attacks)}, attack calibration={len(calibration_attacks)}, "
          f"attack test={len(test_attacks)}, normal test={len(validation_normal) - len(calibration_normals)}")

    print("[2/6] Building normal baseline and calibrating thresholds...")
    baseline = build_baseline_from_entities(
        train_normal, max_exemplars=args.baseline_exemplars, seed=SEED
    )
    normal_pool = [list(e["seq"]) for e in train_normal]
    thresholds = calibrate_all(
        [list(e["seq"]) for e in calibration_normals],
        baseline,
        normal_pool,
        calib_attack_seqs=[list(e["seq"]) for e in calibration_attacks],
        verbose=True,
    )

    print("[3/6] Building labeled reference graph...")
    reference_entities = list(train_normal) + list(training_attacks)
    labels = np.asarray([1 if e["true"] == "attack" else 0 for e in reference_entities], dtype=int)
    for entity in reference_entities:
        entity["D"] = normalized_distance(entity["seq"], baseline)

    alphabet = sorted({str(token) for e in reference_entities for token in e["seq"]}, key=lambda x: int(x))
    rare_tokens = corpus_rare_tokens_real(reference_entities, alphabet, k=15)
    full_features, rare_features = build_features_real(reference_entities, alphabet, rare_tokens)

    print(f"  graph nodes={len(labels)} (normal={int((labels == 0).sum())}, attack={int(labels.sum())})")
    print("[4/6] Computing graph propagation and fitting Stage-1 classifier...")
    adjacency = camouflage_weighted_adjacency(full_features, rare_features, tau=TAU, rare_tau=0.3)
    propagated = precompute_propagation(adjacency, full_features, K=K_PROPAGATION)
    classifier, _ = train_stage1_classifier(
        propagated, rare_features, labels, np.ones(len(labels), dtype=bool), seed=SEED
    )

    # Persist the exact normalization used by train_stage1_classifier.
    stage1_raw_features = np.concatenate([propagated, rare_features], axis=1)
    feature_mean = stage1_raw_features.mean(axis=0)
    feature_scale = stage1_raw_features.std(axis=0) + 1e-6

    print("[5/6] Assembling bundle...")
    bundle = {
        "format": "cascadeguard-tier3-bundle",
        "bundle_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_seed": SEED,
        "dataset": "ADFA-LD",
        "research_status": "experimental_offline_reference_graph",
        "runtime_compatibility": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "config": {
            "tau": TAU,
            "propagation_k": K_PROPAGATION,
            "rare_tau": 0.3,
            "n_chunks": 3,
            "ewma_alpha": 0.5,
            "alphabet": alphabet,
            "rare_tokens": rare_tokens,
            "feature_definition": "normalized token histogram + clipped length/1000 + Tier0 distance; rare histogram auxiliary channel",
            "syscall_ids_are_strings": True,
        },
        "thresholds": {k: float(v) for k, v in thresholds.items()},
        "baseline": [list(s) for s in baseline],
        "normal_pool": normal_pool,
        "reference_full_features": full_features.astype(np.float32),
        "reference_rare_features": rare_features.astype(np.float32),
        "reference_labels": labels,
        "stage1_classifier": classifier,
        "stage1_feature_mean": feature_mean.astype(np.float64),
        "stage1_feature_scale": feature_scale.astype(np.float64),
        "training_summary": {
            "normal_training_traces": len(train_normal),
            "attack_training_traces": len(training_attacks),
            "normal_calibration_traces": len(calibration_normals),
            "attack_calibration_traces": len(calibration_attacks),
            "attack_test_traces_reserved": len(test_attacks),
            "normal_test_traces_reserved": len(validation_normal) - len(calibration_normals),
            "split_seed": SEED,
            "calibration_attack_fraction": args.calibration_attack_fraction,
            "test_attack_fraction": args.test_attack_fraction,
            "baseline_exemplars": len(baseline),
            "reference_graph_nodes": len(labels),
            "threshold_calibration": "labeled G-Means sweep on held-out ADFA-LD normal/attack traces",
        },
    }

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump(bundle, out, compress=3)
    manifest_path = out + ".manifest.json"
    manifest = {
        "format": bundle["format"],
        "bundle_version": bundle["bundle_version"],
        "created_utc": bundle["created_utc"],
        "bundle_file": os.path.basename(out),
        "thresholds": bundle["thresholds"],
        "config": {k: v for k, v in bundle["config"].items() if k not in ("alphabet", "rare_tokens")},
        "training_summary": bundle["training_summary"],
        "research_status": bundle["research_status"],
        "note": "Experimental ADFA-LD research artifact; not a production Kubernetes model validation claim.",
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("[6/6] Export complete")
    print(f"  bundle:   {out}")
    print(f"  manifest: {manifest_path}")
    print("  Next: run evaluate_bundle.py --bundle <path> to evaluate on held-out ADFA-LD traces.")


if __name__ == "__main__":
    main()
