CascadeGuard evaluation diagnostics

Copy diagnose_evaluation.py into model_code/ (same directory as evaluate_bundle.py).
Install pandas if it is not already installed:
  python -m pip install pandas

Run:
  python diagnose_evaluation.py --predictions evaluation/predictions.csv

This reports confusion metrics by decision stage and sweeps candidate thresholds
only for Stage-1 traces. It does not alter the model bundle. Do not select a final
threshold on the test set; use a separate validation/calibration set for that.
