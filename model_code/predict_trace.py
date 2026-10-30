"""Smoke-test the exported CascadeGuard Tier-3 bundle on one syscall sequence.

Example:
  python predict_trace.py --bundle artifacts/cascadeguard_tier3.joblib --syscalls 5 3 3 257 0 1
  python predict_trace.py --bundle artifacts/cascadeguard_tier3.joblib --input-json trace.json

trace.json may contain either ["5", "3", ...] or {"syscalls": ["5", "3", ...]}.
Only syscall-entry events should be converted into a sequence by a future
stream adapter; this CLI accepts an already constructed trace/window.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np

from graph_utils import token_histogram
from tier3_cascade import (
    camouflage_weighted_adjacency,
    gate,
    precompute_propagation,
    stage0_score_real,
)


def predict_sequence(bundle: dict, sequence: list[str]) -> dict:
    seq = [str(x) for x in sequence]
    if len(seq) < 5:
        raise ValueError("Provide at least 5 syscall IDs; use a complete trace/window, not a single event.")

    cfg = bundle["config"]
    baseline = bundle["baseline"]
    normal_pool = bundle["normal_pool"]
    thresholds = bundle["thresholds"]

    # Reuse the research pipeline's Stage-0 score and temporal-memory gate.
    full_d, memory = stage0_score_real(
        seq, baseline, normal_pool,
        n_chunks=int(cfg["n_chunks"]), alpha=float(cfg["ewma_alpha"]),
    )
    gate_decision, gate_score = gate(
        full_d, memory,
        thresholds["L_low"], thresholds["L"],
        thresholds["L_low_mem"], thresholds["L_mem"],
    )

    result = {
        "prediction": None,
        "prediction_label": None,
        "stage0_decision": gate_decision,
        "stage0_distance": float(full_d),
        "temporal_memory": float(memory),
        "gate_score": float(gate_score),
        "sequence_length": len(seq),
        "model_bundle_version": bundle["bundle_version"],
        "research_status": bundle["research_status"],
    }
    if gate_decision == "benign":
        result.update(prediction=0, prediction_label="benign", attack_probability=0.0, decision_stage="stage0")
        return result
    if gate_decision == "malicious":
        result.update(prediction=1, prediction_label="malicious", attack_probability=1.0, decision_stage="stage0")
        return result

    # Append the new trace as a node to the saved reference graph, then rebuild
    # graph propagation. This is exact for this chosen augmented-graph recipe,
    # but O(N^2) and intentionally an experimental baseline, not high-throughput serving.
    alphabet = cfg["alphabet"]
    rare_tokens = cfg["rare_tokens"]
    hist = token_histogram(seq, tokens=alphabet)
    length_norm = min(len(seq) / 1000.0, 1.0)
    full_new = np.concatenate([hist, [length_norm, full_d]])[None, :]
    rare_new = token_histogram(seq, tokens=rare_tokens)[None, :]
    full_all = np.vstack([np.asarray(bundle["reference_full_features"]), full_new])
    rare_all = np.vstack([np.asarray(bundle["reference_rare_features"]), rare_new])

    adjacency = camouflage_weighted_adjacency(
        full_all, rare_all, tau=float(cfg["tau"]), rare_tau=float(cfg["rare_tau"])
    )
    propagated = precompute_propagation(adjacency, full_all, K=int(cfg["propagation_k"]))
    raw = np.concatenate([propagated[-1], rare_new[0]])
    normalized = (raw - bundle["stage1_feature_mean"]) / bundle["stage1_feature_scale"]
    probability = float(bundle["stage1_classifier"].predict_proba(normalized.reshape(1, -1))[0, 1])
    prediction = int(probability >= 0.5)
    result.update(
        prediction=prediction,
        prediction_label="malicious" if prediction else "benign",
        attack_probability=probability,
        decision_stage="stage1_graph_classifier",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experimental CascadeGuard Tier-3 inference on a trace/window")
    parser.add_argument("--bundle", default=os.path.join(os.path.dirname(__file__), "artifacts", "cascadeguard_tier3.joblib"))
    parser.add_argument("--syscalls", nargs="*", help="Syscall IDs in order (one sequence/window)")
    parser.add_argument("--input-json", help="JSON file containing a list or {'syscalls': [...]} ")
    args = parser.parse_args()

    if bool(args.syscalls) == bool(args.input_json):
        parser.error("provide exactly one of --syscalls or --input-json")
    if args.input_json:
        with open(args.input_json, encoding="utf-8") as f:
            payload = json.load(f)
        sequence = payload.get("syscalls") if isinstance(payload, dict) else payload
    else:
        sequence = args.syscalls
    if not isinstance(sequence, list) or not all(isinstance(x, (str, int)) for x in sequence):
        raise SystemExit("Input must be a list of syscall IDs or an object with a 'syscalls' list")

    bundle = joblib.load(args.bundle)
    if bundle.get("format") != "cascadeguard-tier3-bundle":
        raise SystemExit("Unrecognized bundle format")
    print(json.dumps(predict_sequence(bundle, sequence), indent=2))


if __name__ == "__main__":
    main()
