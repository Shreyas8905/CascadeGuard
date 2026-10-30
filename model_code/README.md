# Cascade-Guard Ablation — Real-Data (ADFA-LD) Version

This supersedes the earlier synthetic-data package from this conversation.
It runs the same three-tier ablation (ShadowKube-only / vanilla-GuardNet
bolt-on / Cascade-Guard) against **real, labeled intrusion-detection data**
instead of a generated stand-in.

## Dataset: ADFA-LD (Creech & Hu)

The real LID-DS dataset discussed earlier in this conversation is hosted on
`drive.proton.me`, which is not reachable from the sandbox this was built
in. **ADFA-LD** is used instead — a real, peer-reviewed, still widely-cited
Linux host-intrusion-detection benchmark, with a labeled mirror hosted
directly on GitHub (which *is* reachable):

- **Labeled mirror (what this package uses):**
  `https://github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset`
- **Original dataset source (UNSW Canberra Cyber):**
  `https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-IDS-Datasets/`
- **Citations (please cite these, not just the GitHub mirror, if you use this data):**
  1. G. Creech and J. Hu, *A Semantic Approach to Host-based Intrusion
     Detection Systems Using Contiguous and Discontiguous System Call
     Patterns*, IEEE Transactions on Computers, 2013.
  2. G. Creech and J. Hu, *Generation of a new IDS test dataset: Time to
     retire the KDD collection*, IEEE WCNC 2013.
  3. G. Creech, *Developing a high-accuracy cross platform Host-Based
     Intrusion Detection System capable of reliably detecting zero-day
     attacks*, PhD thesis, UNSW, 2014.

### Downloading it yourself

```bash
git clone https://github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset.git adfa_ld
cd adfa_ld
unzip ADFA-LD.zip -d extracted
```

After this, you should have:
```
adfa_ld/extracted/ADFA-LD/Training_Data_Master/*.txt      (833 normal traces)
adfa_ld/extracted/ADFA-LD/Validation_Data_Master/*.txt    (4372 normal traces)
adfa_ld/extracted/ADFA-LD/Attack_Data_Master/<Cat>_<n>/*.txt  (746 attack traces,
    6 categories: Adduser, Hydra_FTP, Hydra_SSH, Java_Meterpreter, Meterpreter, Web_Shell)
```
`load_adfa.py`'s `DEFAULT_ROOT` expects the `adfa_ld/` folder to sit next to
the Python files (i.e. `./adfa_ld/extracted/ADFA-LD/...`) — either place it
there or pass a different `root=` to `load_adfa()`.

Each `.txt` file is one whitespace-separated line of real Linux syscall
numbers — one full session/trace per file, no thread/pod structure (each
trace is treated as one independent entity, matching ShadowKube's own
"one container per pod" simplification).

## Files

| File | Role |
|---|---|
| `load_adfa.py` | Parses the real dataset into normal-train / normal-test / attack entity lists |
| `shadowkube_tier1.py` | Tier 1: ShadowKube's LCS/Levenshtein scoring rule (Eq. 4), against a real-trace baseline pool |
| `mimicry.py` | Mimicry-attack construction: dilutes a real attack trace inside real, resampled normal material (Wagner & Soto, 2002) |
| `ngram_utils.py` | Generic n-gram frequency counter (filler-bank builder for mimicry) |
| `graph_utils.py` | Feature vectors (full profile + rare-token profile) + cosine-similarity graph construction (GuardNet Eq. 1) |
| `tier2_gcn.py` | Tier 2: hand-rolled 2-layer GCN, GuardNet-style MCD downsampling + class-weighted loss |
| `tier3_cascade.py` | Tier 3: Cascade-Guard — accelerated Stage 0, confidence-band gate, EWMA temporal memory, camouflage-resistant SGC-style Stage 1 |
| `calibration.py` | Derives L / L_low / L_mem / L_low_mem from a genuine labeled G-Means sweep on held-out real data |
| `run_ablation_real.py` | Main entry point |
| `scaling_check.py` | Standalone O(N²) graph-construction scaling confirmation (synthetic feature vectors, architecture-level check, not data-dependent) |
| `requirements.txt` | Exact package versions used |

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# then download the dataset as shown above, into ./adfa_ld/
```

## Run

```bash
python3 run_ablation_real.py
```

Expect this to take **a few minutes**, not seconds — real Levenshtein
comparisons against a real baseline pool, across ~1,300+ evaluation
entities and two conditions, plus the mimicry-construction search loop
(labeled ~1-3 min in the script's own progress output) are real work, unlike
the synthetic version's sub-5-second runtime. `rapidfuzz` (a compiled
Levenshtein implementation) is required for this to be tractable at all —
pure-Python edit distance was ~500x slower in early testing on this dataset.

Everything is seeded (`SEED = 42` in `run_ablation_real.py`), so results
should reproduce on repeated runs given identical package versions — the
one exception is `p50_ms`/`p95_ms` latency figures, which measure your CPU
and will naturally vary run to run and machine to machine; the accuracy
metrics (recall/F1/G-Means/TP/FP counts) should not.

## What to expect — an honest heads-up before you run this

This real-data pass surfaced findings that are **more interesting, and less
flattering to ShadowKube's baseline detector, than the earlier synthetic
run was**. Specifically, in development testing of this exact code:

1. **Real ADFA-LD's normal and attack score distributions overlap far more
   than the synthetic corpus's did.** A first attempt calibrated the
   detection threshold purely from held-out normal traffic (targeting a
   fixed false-positive rate) and got **2% recall** on genuine, unmodified
   attacks — the threshold needed to keep FPR low on real data turned out
   to sit almost entirely above where real attacks score too. This is why
   `calibration.py` instead does a genuine **labeled sweep** (using a small
   held-out slice of real attack traces, carved out before any mimicry
   construction) to find the G-Means-optimal threshold. At that operating
   point, expect Tier 1's *own, unmodified-attack* performance to already
   look meaningfully weaker than ShadowKube's own reported ~97% TP rate —
   real system behavior is messier than a curated CVE-exploitation
   benchmark, and this is a legitimate, reportable limitation in its own
   right, independent of the mimicry-attack story.
2. **Escalation rate in Cascade-Guard's gate may run high on real data**
   (we saw 80%+ in testing) — a direct, honest consequence of (1): if
   normal and attack scores overlap this much, a *narrow* confidently-benign
   zone is a correct reflection of the data, not a bug. This is worth
   reporting as-is rather than papering over by artificially widening the
   gate band to hit a specific target escalation rate.
3. Given (1) and (2), don't be surprised if Tier 2/3's relative advantage
   over Tier 1 on the **mimicked** condition looks different in degree than
   the earlier synthetic run's headline numbers — the synthetic corpus was,
   by construction, cleaner and more separable than real system telemetry.

None of this invalidates the core architectural argument (graph-based
relational reasoning should recover mimicry-induced recall loss); it just
means the *magnitude* of every number here should be read as "real ADFA-LD,
one specific real HIDS benchmark," not as a stand-in for how this would
perform against ShadowKube's actual target environment (Kubernetes pod
behavior), which remains a real gap — see the study report's Section on
Kubernetes-specific data collection as necessary follow-up.

## Known limitations of this pass

1. **No pod/namespace grouping.** ADFA-LD has no per-scenario metadata the
   way a Kubernetes deployment would, so there is one global baseline
   (`load_adfa.GLOBAL_GROUP`), not per-group baselines as ShadowKube itself
   uses. Attack category (Adduser/Hydra_FTP/etc.) is tracked for stratified
   reporting only and deliberately excluded from graph features (it would
   otherwise hand the label to the classifier).
2. **MCES (subgraph fusion) is not separately implemented in Tier 2** — see
   `tier2_gcn.py`'s docstring for why, and revisit once graphs are built
   from a real, larger, more disconnected Kubernetes cluster.
3. **Absolute latency numbers are implementation-speed artifacts, not
   ShadowKube-comparable wall-clock figures** — see `run_ablation_real.py`'s
   docstring for the exact accounting rules (what's charged to per-event
   latency vs. treated as offline/periodic). The *relative* ordering and
   the O(N²) graph-construction scaling trend (confirmed in
   `scaling_check.py`) are the meaningful outputs, not the millisecond
   values themselves at this entity count.
4. **Temporal-memory (EWMA) calibration**, like the full-score threshold,
   now comes from a real labeled sweep rather than an arbitrary multiplier
   — a real improvement over an earlier draft of this pipeline, which used
   undocumented constants (`L_low * 1.4`, `L * 1.15`) that had no empirical
   basis and produced a degenerate near-100%-escalation gate. If you change
   `N_CALIB_NORMAL` / `CALIB_ATTACK_FRAC` / the dataset itself, re-run to
   get freshly calibrated thresholds rather than reusing old ones.

## Export a reusable Tier-3 bundle (added)

The original `run_ablation_real.py` is an **offline ablation experiment**; it does not export a serving model. The new `export_tier3.py` trains a reusable experimental reference-graph bundle and saves it to `artifacts/cascadeguard_tier3.joblib`, with a JSON manifest alongside it.

From this directory (`files - Copy/`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python export_tier3.py
```

The export command:

1. Loads the local ADFA-LD traces.
2. Builds the Tier-0 normal baseline from `Training_Data_Master`.
3. Holds out normal and attack traces for threshold calibration.
4. Calibrates the four Tier-0/gate thresholds using `calibration.py`.
5. Trains the Stage-1 classifier over a labeled reference graph and saves the graph feature references, classifier, normalization statistics, thresholds, baseline and feature configuration.

To smoke-test loading and inference on an example sequence/window:

```bash
python predict_trace.py --bundle artifacts/cascadeguard_tier3.joblib --syscalls 5 3 3 257 0 1 5 3 3 1
```

You can also pass a JSON list or `{"syscalls": ["5", "3", "3", "257"]}` using `--input-json trace.json`. Input must be a sequence/window, not one isolated event.

### Important limitations

- The bundle is a **research/experimental artifact trained and calibrated on ADFA-LD**, not a validated Kubernetes production model.
- `predict_trace.py` demonstrates sequence-level inference. It does not yet turn the collector's event stream into per-process windows, persist temporal memory across windows, or expose an HTTP API.
- The Stage-1 path augments the saved reference graph with each incoming trace and recomputes propagation. This is deliberately straightforward and **O(N²)**; it is not a high-throughput serving design.
- ADFA-LD syscall IDs come from an older Linux environment. Before applying this model to current eBPF events, verify the syscall number/architecture mapping and quantify dataset/domain shift. A successful prediction is not evidence of production detection quality.
- `joblib` artifacts should only be loaded from trusted sources because pickle-based formats can execute code during deserialization.
- The original ablation runner and its results remain intact. Export training is a separate path so you can reproduce the experiment independently from the serving bundle.

## Leakage-safe held-out evaluation

The exporter reserves separate attack traces for threshold calibration and final testing. Validation-normal traces are also split into calibration and held-out test subsets. Test traces are excluded from the Stage-1 reference graph and threshold calibration.

Re-export the bundle after updating the code:

```bash
python export_tier3.py
```

Then evaluate it:

```bash
python evaluate_bundle.py --bundle artifacts/cascadeguard_tier3.joblib
```

Outputs are written to `evaluation/metrics.json` and `evaluation/predictions.csv`. For a quick smoke test, use `--max-traces 20` (per class); use the full test set for reported results. Metrics are ADFA-LD offline results and must not be presented as Kubernetes detection performance.
