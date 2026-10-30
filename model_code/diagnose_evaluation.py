#!/usr/bin/env python3
"""Diagnose CascadeGuard evaluation outputs without retraining the model.

Reads evaluation/predictions.csv and reports decisions by stage, then sweeps
Stage-1 thresholds ONLY among traces actually escalated to Stage 1. It does not
change the exported bundle or claim a new operating threshold is validated.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if tn + fp else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', default='evaluation/predictions.csv')
    ap.add_argument('--output', default='evaluation/diagnostics.json')
    ap.add_argument('--thresholds', default='0.30,0.40,0.50,0.60,0.70,0.80,0.90')
    args = ap.parse_args()
    df = pd.read_csv(args.predictions)
    required = {'true_label','prediction','attack_probability','stage0_decision','decision_stage'}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'Missing columns: {sorted(missing)}')
    y = (df.true_label == 'attack').astype(int).to_numpy()
    pred = (df.prediction == 'malicious').astype(int).to_numpy()
    out = {'n_total': int(len(df)), 'overall': metrics(y, pred), 'stage_counts': {}, 'stage_metrics': {}, 'stage1_threshold_sweep': []}
    for col in ['stage0_decision', 'decision_stage']:
        out['stage_counts'][col] = {str(k): int(v) for k, v in df[col].value_counts(dropna=False).items()}
    for stage, part in df.groupby('decision_stage'):
        idx = part.index.to_numpy()
        yy = (part.true_label == 'attack').astype(int).to_numpy()
        pp = (part.prediction == 'malicious').astype(int).to_numpy()
        out['stage_metrics'][str(stage)] = {'n': int(len(part)), **metrics(yy, pp),
            'true_normal': int((yy == 0).sum()), 'true_attack': int((yy == 1).sum())}
    s1 = df[df.decision_stage == 'stage1_graph_classifier'].copy()
    if len(s1):
        ys = (s1.true_label == 'attack').astype(int).to_numpy()
        scores = pd.to_numeric(s1.attack_probability).to_numpy()
        for t in [float(x) for x in args.thresholds.split(',') if x.strip()]:
            pp = (scores >= t).astype(int)
            out['stage1_threshold_sweep'].append({'threshold': t, 'n_escalated': int(len(s1)), **metrics(ys, pp)})
    else:
        out['stage1_note'] = 'No traces reached Stage 1; threshold sweep unavailable.'
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
    print(f'\nSaved: {Path(args.output).resolve()}')

if __name__ == '__main__':
    main()
