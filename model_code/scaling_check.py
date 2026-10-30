"""
Standalone scaling check: confirms the O(N^2 d) graph-construction cost
(GuardNet's own stated complexity, and ~40% of their measured runtime on the
real 284,807-row ccfraud dataset, their Table 4/5) actually grows
quadratically here too. This is what motivates Cascade-Guard's decision to
treat graph construction as an offline/periodic cost rather than a per-
window online one (see tier3_cascade.py's module docstring).

Run: python3 scaling_check.py
"""
import time
import numpy as np
from graph_utils import build_adjacency

if __name__ == "__main__":
    print(f"{'N':>8} | {'time (s)':>10} | {'per-node (ms)':>14}")
    for N in [500, 2000, 8000, 12000]:
        X = np.random.default_rng(0).normal(size=(N, 45)).astype(np.float32)
        t0 = time.perf_counter()
        build_adjacency(X, tau=0.97)
        t1 = time.perf_counter()
        print(f"{N:8d} | {t1 - t0:10.4f} | {(t1 - t0) / N * 1000:14.5f}")
