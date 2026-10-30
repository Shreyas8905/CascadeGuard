"""Evaluate an exported CascadeGuard bundle on deterministic held-out ADFA-LD traces.

The split mirrors export_tier3.py: attack traces are partitioned into threshold-
calibration, test, and training sets; validation normals are split into threshold-
calibration and test sets. Test traces are never added to the reference graph.

Example:
  python evaluate_bundle.py --bundle artifacts/cascadeguard_tier3.joblib
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

from load_adfa import load_adfa
from predict_trace import predict_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CascadeGuard on held-out ADFA-LD traces")
    parser.add_argument("--bundle", default=os.path.join(os.path.dirname(__file__), "artifacts", "cascadeguard_tier3.joblib"))
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "evaluation"))
    parser.add_argument("--max-traces", type=int, default=0, help="Optional cap per class for quick smoke tests; 0 evaluates all")
    args = parser.parse_args()

    bundle = joblib.load(args.bundle)
    if bundle.get("format") != "cascadeguard-tier3-bundle":
        raise SystemExit("Unrecognized bundle format")
    summary = bundle.get("training_summary", {})
    if "attack_test_traces_reserved" not in summary:
        raise SystemExit("This bundle predates the leakage-safe test split. Re-run export_tier3.py first.")

    print("[1/4] Loading ADFA-LD and reproducing held-out split...")
    train_normal, validation_normal, attacks = load_adfa()
    seed = int(summary.get("split_seed", 42))
    rng = np.random.default_rng(seed)
    attack_order = rng.permutation(len(attacks))
    n_cal_attack = int(summary["attack_calibration_traces"])
    n_test_attack = int(summary["attack_test_traces_reserved"])
    test_attack_idx = attack_order[n_cal_attack:n_cal_attack + n_test_attack]
    normal_order = rng.permutation(len(validation_normal))
    n_cal_normal = int(summary["normal_calibration_traces"])
    test_normal_idx = normal_order[n_cal_normal:]

    test_normal = [validation_normal[i] for i in test_normal_idx]
    test_attacks = [attacks[i] for i in test_attack_idx]
    if args.max_traces > 0:
        test_normal = test_normal[:args.max_traces]
        test_attacks = test_attacks[:args.max_traces]
    test_items = [("normal", 0, e) for e in test_normal] + [("attack", 1, e) for e in test_attacks]
    random.Random(seed).shuffle(test_items)
    print(f"  held-out normal={len(test_normal)}, held-out attack={len(test_attacks)}")

    print("[2/4] Running cascade inference...")
    rows = []
    y_true, y_pred, y_score = [], [], []
    for i, (kind, label, entity) in enumerate(test_items, 1):
        seq = [str(x) for x in entity["seq"]]
        try:
            result = predict_sequence(bundle, seq)
            pred = int(result["prediction"])
            score = float(result.get("attack_probability", pred))
            rows.append({
                "trace_index": i,
                "true_label": kind,
                "prediction": result["prediction_label"],
                "attack_probability": score,
                "stage0_decision": result["stage0_decision"],
                "stage0_distance": result["stage0_distance"],
                "temporal_memory": result["temporal_memory"],
                "decision_stage": result["decision_stage"],
                "sequence_length": len(seq),
            })
            y_true.append(label)
            y_pred.append(pred)
            y_score.append(score)
        except Exception as exc:
            raise RuntimeError(f"Inference failed for held-out {kind} trace #{i}: {exc}") from exc
        if i % 50 == 0 or i == len(test_items):
            print(f"  processed {i}/{len(test_items)}")

    print("[3/4] Computing metrics...")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    metrics = {
        "evaluation_time_utc": datetime.now(timezone.utc).isoformat(),
        "bundle": os.path.abspath(args.bundle),
        "dataset": "ADFA-LD",
        "evaluation_protocol": "held-out attack traces and held-out validation-normal traces; test traces excluded from reference graph and threshold calibration",
        "n_test": len(y_true),
        "n_normal": int(sum(1 for y in y_true if y == 0)),
        "n_attack": int(sum(1 for y in y_true if y == 1)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_attack": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_attack": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_attack": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"labels": ["normal", "attack"], "matrix": cm.tolist(), "tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else None,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else None,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) == 2 else None,
        "pr_auc_average_precision": float(average_precision_score(y_true, y_score)) if len(set(y_true)) == 2 else None,
        "classification_report": classification_report(y_true, y_pred, target_names=["normal", "attack"], output_dict=True, zero_division=0),
        "limitations": [
            "ADFA-LD is not Kubernetes workload telemetry; results do not establish Kubernetes intrusion-detection performance.",
            "The inference path uses an experimental augmented reference graph and recomputes graph propagation per escalated trace.",
            "Attack probability is a model score; calibration on this dataset does not guarantee calibrated operational risk.",
        ],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "metrics.json")
    csv_path = os.path.join(args.output_dir, "predictions.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print("[4/4] Evaluation complete")
    print(json.dumps({k: metrics[k] for k in ("n_test", "n_normal", "n_attack", "accuracy", "precision_attack", "recall_attack", "f1_attack", "roc_auc", "pr_auc_average_precision", "confusion_matrix", "specificity", "false_positive_rate", "false_negative_rate")}, indent=2))
    print(f"  metrics:     {os.path.abspath(json_path)}")
    print(f"  predictions: {os.path.abspath(csv_path)}")


if __name__ == "__main__":
    main()
