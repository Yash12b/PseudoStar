# Final AI/ML Audit — every model, measured evidence only

Date: 2026-09-16. Method: checkpoints loaded and compared against fresh
inits; training/eval code read; artifact metadata cross-checked; runtime
wiring traced. No ML-superiority claim is made without a measured number
below. Allowed provenance: `trained | random | imported |
deterministic_baseline`.

---

## 1. Temporal GRU predictor (temporal-v1 .. v4)

- NAME: Temporal GRU displacement predictor (`temporal_gru.pt`)
- PROVENANCE: **trained** (weights moved from init, per-tensor max-diff
  0.03–0.12 vs init scale ~0.10)
- DATASET: closed-loop sim rollouts, 10 s @ 30 fps, trajectories
  straight/circular/figure-8/random, disturbances clear+light
  (`ai/temporal_dataset.py`). temporal-v3 manifest: train 22 seq /
  6600 samples, val 4 / 1200, test 6 / 1800, hard_test 6 / 1800.
  Splits are same-config shuffles differing only in seed (no cross-type
  generalization). `hard_test` "fast" variants never run (variant-drop
  bug `:443-444`): hard_test is distributionally identical to train.
- TRAINED?: yes, but the **uncertainty head is bit-identical across all
  four checkpoints (max-diff 0.0)** — it is never in the loss graph
  (`temporal_training.py:109-112` uses displacement only). Uncertainty
  outputs are effectively **random**.
- ARCHITECTURE: 1-layer GRU(15→32) + Linear(32→10) displacement head +
  Linear(32→10) uncertainty head; 5,364 params
  (`ai/neural.py:94-129`).
- TRAIN/VAL/TEST: v4 metadata: 22 / 4 seqs, best_val_loss 209.45
  (honest, post-fix). v3 best_val 0.000233 is the documented **fake**
  pre-fix value (last-frame zero-displacement labels).
- METRICS: GRU **~35.16 px RMSE** vs constant-velocity **~4.51 px**
  (`temporal_evaluation.py:39-61,95-116`). Thin protocol: ~6 test
  sequences x 1 mid-sequence timestep, no seed averaging, **no
  shipped metrics.json** — numbers live only in this doc lineage.
- BASELINE: constant velocity wins. **No ML superiority; CV stays.**
- RUNTIME USE: **none by default.** `RuntimePredictor`/`AIIntegration`
  default to CV (`GRUPredictor.available=False`). A supplied checkpoint
  would additionally hit: raw-state-dict reject (fixed — see §12),
  hardcoded `hidden_dim=32`, training-vs-runtime **feature-order
  mismatch**, and 1-step vs 300-step distribution shift.
- LATENCY: measured CPU forward: 1.28 ms (300-step) / 0.019 ms (1-step).
- STATUS: **TRAINED / EXPERIMENTAL — quarantined from production.**
- LIMITATIONS: 22 training seqs; uncertainty head untrained; tail-label
  fallbacks (last ~15 frames) invalid for full-sequence eval; timestamp
  feature leaks position-in-sequence; failure labels use future LOST
  (offline-only, never inputs).

## 1b. Temporal GRU v5 (2026-09-17 retrain — more epochs, fresh seed)

- Real run: temporal-v3 data (22 train seqs), 150-epoch budget, seed 123,
  early-stopped epoch 14, best val 214.4 (≈ v4's 209.5). New versioned
  dir `artifacts/models/temporal-v5/`; v4 untouched.
- Held-out test (6 seqs × t=150, same protocol): GRU-v5 **35.72 px** vs
  CV **4.51 px** vs last-position 36.03 px (≈ v4's 35.16 px). Five times
  the epoch budget yields nothing: with 22 sequences the GRU cannot
  even separate from last-position, let alone CV.
- STATUS: **TRAINED / EXPERIMENTAL — not promoted, correctly.**
- NOTE: genuine promotion needs ~10× data (≈9 h generation, costed,
  not scheduled) and possibly a better formulation than displacement
  regression on velocity-redundant features.

## 2. Heavy trajectory-GRU (heavy-training-v1)

- NAME: trajectory-gru (`model.pt`, 42,757 params)
- PROVENANCE: **trained** (train/val loss 6.1e-6/6.5e-6 — suspiciously
  near-zero on normalized synthetic targets; likely memorization)
- DATASET: pure NumPy RNG (`(3000,30,15)` sequences, `heavy_training.py:
  129-210`), NOT the pipeline-observable distribution. No test split.
- TRAINED?: yes. ARCHITECTURE: 2-layer GRU(15→64) + MLP head → 5-vector
  `[dx,dy,vx,vy,will_lose]` — **API-incompatible** with the
  multi-horizon consumer. TRAIN/VAL/TEST: train/val only.
- METRICS: val loss only; no RMSE, no baseline comparison.
- BASELINE: none. RUNTIME USE: **none; unloadable** (hidden_dim/head
  mismatch). LATENCY: not measured (cannot load into runtime path).
- STATUS: **TRAINED / UNUSED.** LIMITATIONS: synthetic-only, no test
  metrics, incompatible output API.

## 3. Constant-velocity predictor

- NAME: `ConstantVelocityPredictor` (cv-v1)
- PROVENANCE: **deterministic_baseline** (analytical; `trained=False`)
- DATASET: none. TRAINED?: no. ARCHITECTURE: `pred = vel * horizon`,
  linear uncertainty growth.
- TRAIN/VAL/TEST: n/a. METRICS: **~4.51 px RMSE** (same thin protocol
  caveat as §1). BASELINE: it IS the baseline — and it beats the GRU.
- RUNTIME USE: **production default** in `RuntimePredictor` and all
  tests; gated by 3×-CV sanity check + `SafetyGate` when GRU attempted.
- LATENCY: microseconds (closed-form). STATUS: **PRODUCTION.**
- LIMITATIONS: no maneuver handling; uncertainty growth is a fixed rate.

## 4. Expert Policy

- NAME: `ExpertPolicy` (`ai/mission.py:356-394`)
- PROVENANCE: **deterministic_baseline** (pure if/elif on situation →
  `MissionAction`, always via `SafetyEnvelope.approve`)
- DATASET: none. TRAINED?: no. ARCHITECTURE: rules.
- TRAIN/VAL/TEST: n/a. METRICS: outcome metrics come from closed-loop
  benchmarks, not the policy itself. BASELINE: it IS the baseline.
- RUNTIME USE: **production default** (`AIMissionBrain` default, GUI
  worker, `FULL_AI_MISSION` fallback).
- LATENCY: microseconds. STATUS: **PRODUCTION.**
- LIMITATIONS: no learning; `Situation.RECOVERY` branch unreachable
  (classifier never emits it — dead code, `mission.py:119-169`).

## 5. Learned Policy (mission-v1/v2/v3 `.npz`)

- NAME: `LearnedFeatureClassifier` policy head, ridge multinomial-linear
- PROVENANCE: **trained** (closed-form `solve`, real `(15,11)` weights;
  mission-v3 `policy-v2.npz` labels == all 11 current `MissionAction`s)
- DATASET: observable sim features; mission-v3: train 6600 / val 1200 /
  test 1800. TRAINED?: yes.
- ARCHITECTURE: linear softmax. TRAIN/VAL/TEST: mission-v3 policy
  0.9895 / 0.965 / **0.9894 test**.
- METRICS: ~98.9% test accuracy — but this is **teacher-mimicry**
  (labels ARE the expert's outputs on the same inputs;
  `train_mission_real.py:74-89`). Situation/policy accuracies identical
  to 4 decimals because the expert is a deterministic function of the
  situation. **Not an outcome-quality metric.**
- BASELINE: expert (the teacher). No superiority claim possible by
  construction — it distills the baseline.
- RUNTIME USE: **gated production**: benchmark loads policy weights
  only if labels exactly equal the current `MissionAction` set
  (`benchmark/methods.py:114-136`); else NOT AVAILABLE, never
  fabricated. GUI default remains expert-only.
- LATENCY: microseconds (15×11 matvec). STATUS: **TRAINED /
  PRODUCTION-GATED (behavioral clone).**
- LIMITATIONS: mimicry ceiling = teacher; mission-v1 motion RMSE ~6e-10
  exposes that v1 data was trivially synthetic (superseded by v2/v3).

## 6. Heavy control-policy (heavy-training-v1)

- NAME: control-policy (`model.pt`, GRU 15-dim → pan/tilt/confidence)
- PROVENANCE: **trained** (best_val 1.206, 50 epochs) on hand-coded PID
  labels — a second, separate teacher-mimicry.
- DATASET: synthetic. TRAIN/VAL/TEST: train/val only.
- METRICS: val loss only. BASELINE: PID teacher. RUNTIME USE: **none.**
  `train_mission_models()` (`heavy_training.py:561-611`) is dead/broken
  (`TypeError` on missing labels arg).
- LATENCY: not wired. STATUS: **TRAINED / DEAD CODE.**
- LIMITATIONS: unused; broken training entry point; no test metrics.

## 7. Situation rule classifier (production)

- NAME: `SituationClassifier` (`ai/mission.py:100-169`)
- PROVENANCE: **deterministic_baseline** (threshold logic on observable
  features only — no GT inputs)
- DATASET: none. TRAINED?: no. ARCHITECTURE: rules, 10 outputs
  (NORMAL_TRACKING … RECOVERY).
- TRAIN/VAL/TEST: n/a. METRICS: n/a (production path).
- RUNTIME USE: **production** — always computed in
  `AIMissionBrain.decide`. LATENCY: microseconds.
- STATUS: **PRODUCTION.** LIMITATIONS: never emits RECOVERY (see §4).

## 8. Learned situation weights (mission-*/situation-*.npz)

- NAME: situation linear head. PROVENANCE: **trained** — on a
  **stale 7-label set** (`normal_tracking…anomaly`, incl.
  `processing_overload`), shape (15,7) in ALL versions.
- DATASET: older observable data. TRAINED?: yes, but obsolete.
- METRICS: "98.9% (7 classes)" describes the stale artifact's mimicry
  score, not production. BASELINE: rule classifier.
- RUNTIME USE: **none — quarantined.** No loader offered; exact-label
  validation would refuse it (`benchmark/methods.py:10-12`).
- LATENCY: n/a. STATUS: **TRAINED / STALE / QUARANTINED.**
- LIMITATIONS: label set predates enum expansion; current
  `train_mission_real.py` trains 10 labels, so these files could not
  have been produced by the current script.

## 9. Visual beacon CNN (beacon-visual-v1)

- NAME: `TinyVisualBeaconNet` (`checkpoint.pt`, 14,276 params)- PROVENANCE: **trained** (conv weights moved from He init, L2 3–5)
- DATASET: synthetic only (`ai/dataset.py`): train 1000 (804 visible) /
  val 200 / test 200 / hard_test 200, seed 42, 128×128.
- TRAINED?: yes (10 epochs, best_val_loss 1.133; heatmap BCE w=8 +
  presence BCE w=4). ARCHITECTURE: tiny conv net → heatmap + presence.
- TRAIN/VAL/TEST: 1000/200/200(+200 hard). METRICS: test P 0.81 / R
  0.60 / F1 0.69 / FPR 0.46 / RMSE 168 px; hard P 0.25 / R 0.28 /
  FPR 1.0. **Poor localization/generalization.**
- BASELINE: classical detector (authoritative; visual not enabled until
  it wins — `docs/ai/visual-perception.md`, consistent).
- RUNTIME USE: **none.** `.pt` loads only in offline train/eval
  scripts; runtime `BeaconCNN` is a separate NumPy model, `random` by
  default. LATENCY: not production-relevant.
- STATUS: **TRAINED / EXPERIMENTAL — not wired.**
- LIMITATIONS: synthetic-only; eval size default (64) mismatches
  training size (128) unless overridden; hard_test FPR 1.0.

## 9b. Visual beacon CNN v2 (beacon-visual-v2, 2026-09-17 retrain)

- NAME: `TinyVisualBeaconNet`, same architecture, fresh 3000-sample
  dataset (seed 2026), 5 epochs, best val loss 1.050 (vs v1 1.133).
- PROVENANCE: **trained** (real run, versioned receipt in
  `artifacts/models/beacon-visual-v2/`; v1 untouched).
- METRICS on beacon-v3 test (same images for both): v2 P/R/F1
  0.13/0.13/**0.21**, RMSE 216 px; v1 on v3: F1 0.43, RMSE 279 px.
  Classical detector on noisy video: 0.37 px. **The CNN loses by two
  orders of magnitude — an architecture/formulation gap, not a data
  shortage. Not promoted.**
- STATUS: **TRAINED / EXPERIMENTAL — not wired.**
- NOTE: a GRU retrain was scoped but rejected in-session: temporal
  dataset generation costs ~160 s/sequence (≈9 h for 200 sequences).
  Documented as future work with honest costing, not faked.

## 10. NumPy BeaconCNN (runtime visual model)

- NAME: `BeaconCNN` (`ai/model.py`), NumPy inference (`ai/inference.py`)
- PROVENANCE: **trained** (`ai/train_beacon_cnn.py`: torch replica of
  the exact numpy architecture, 4000 synthetic samples, 40 epochs,
  exported weight-for-weight to `artifacts/models/beacon-numpy-v1/`,
  parity 6e-06, receipt with metrics).
- DATASET: synthetic (`ai/dataset.py`, seed 42). METRICS: 17/20 hits
  @~6 px synthetic probes, 0 FP on empties; 98.8% detection but
  297 px RMSE on the noisy 640×480 reference (fires on noise
  structure) vs 0.32 px classical. BASELINE: classical detector.
- RUNTIME USE: opt-in `ai`/`hybrid` backends with trained-or-fallback
  contract (missing weights → classical + warning); vectorized
  conv/pool + bilinear resize (≈3 ms steady-state).
- STATUS: **TRAINED / EXPERIMENTAL.** LIMITATIONS: coarse 1/8
  heatmap localization; noise-fragile; classical default retained.

## 11. Disturbance MLPs (disturbance-v1: classifier / controller / predictor)

- NAME: 3 MLPs (12→64→64→7; 19→64→64→6+Softplus; GRU 12→64×2+FC)
- PROVENANCE: **trained** (L2 vs fresh init 138/144/1747; clearly
  optimized). DATASET: synthetic hand-coded (`n=5000, seq_len=10`,
  seed 42), single 80/20 split, **no test split, no split seed**.
- TRAINED?: yes. METRICS: val loss/accuracy only; fog sanity check
  post-hoc. BASELINE: none.
- RUNTIME USE: **none — dead code.** No runtime import outside
  `disturbance_training.py`. LATENCY: n/a.
- STATUS: **TRAINED / UNUSED.** LIMITATIONS: synthetic-only, no test
  metrics, unseeded split.

## 12. Failure predictor (+ failure-v1.npz)

- NAME: ridge logistic regression on 60-dim window features
- PROVENANCE: **trained** (`failure-v1.npz` + experiment.json with
  train-acc/val-F1/test-F1 from `train_failure.py`)
- DATASET: `temporal-v1` features/masks/failure-labels, window 10.
- TRAINED?: artifact yes. RUNTIME USE: runtime instantiations in
  `gui/worker.py`, `ai_benchmark.py`, `runner.py` are **untrained**
  (heuristic `risk=0` fallback); no `.load()` call exists in `src/`.
- LATENCY: microseconds. STATUS: **TRAINED ARTIFACT / NOT LOADED —
  effective provenance at runtime: random/heuristic.**
- LIMITATIONS: failure labels derive from future LOST (offline-only);
  temporal-v1 had ~0% failure rate (Kalman never formally LOST),
  so the supervision signal is near-degenerate.

---

## Leakage & validity audit (all components)

- Label leakage: policy/situation accuracies are teacher-mimicry BY
  DESIGN (disclosed in `policy-model.md`, `situation-model.md`, and
  above). No outcome claim is built on them.
- Future-state leakage: temporal labels use future tracker estimates
  (legitimate offline supervision); failure labels use future LOST
  (offline-only). No future values in any runtime input — verified
  feature lists (`temporal_dataset.py:133-173`, `mission.py:48-49`).
- Simulator-truth leakage: temporal features are tracker/clock-derived
  only; policy/situation use `ObservationFeatures` (no GT). Clean.
- Duplicate samples: temporal configs collapse to seed-only variants
  (reduced diversity, not duplication); hard_test duplicates train
  distribution (variant-drop bug) — documented, not hidden.
- Train/test contamination: `fit` touches train only; `evaluate` never
  refits. Splits are separate sim runs. No test-set training found.
- Invalid timesteps: pre-fix last-frame zero-label bug documented and
  fixed (v4 + `T-16` guards); v3 `best_val 0.000233` preserved as the
  fake reference, not a claim.
- Fake metrics: one historical fake (pre-fix 0.035px / "837× better")
  already retracted in acceptance-report. Current numbers are
  measured-but-thin (single-timestep, ~6 seqs, unshipped JSON) —
  stated as such above, never as superiority claims.
- **Code defect fixed in this audit:** `GRUPredictor._try_load` marked
  unloadable checkpoints (raw state dicts incl. all `temporal-v*` files)
  as `trained=True/available=True` while keeping random weights.
  Now rejected → deterministic fallback (`runtime_predictor.py:175+`).
  `ConstantVelocityPredictor.provenance.trained` corrected True→False.

## Provenance map (one line each)

| Model | Provenance | Status |
|---|---|---|
| Temporal GRU v1–v4 | trained | EXPERIMENTAL, quarantined |
| Heavy trajectory-gru | trained | UNUSED (incompatible API) |
| Constant velocity | deterministic_baseline | PRODUCTION |
| Expert Policy | deterministic_baseline | PRODUCTION |
| Learned Policy v1/v2/v3 | trained (clone) | PRODUCTION-GATED |
| Heavy control-policy | trained (clone) | DEAD CODE |
| Situation rules | deterministic_baseline | PRODUCTION |
| Learned situation npz | trained | STALE, QUARANTINED |
| Visual TinyVisualBeaconNet | trained | EXPERIMENTAL, not wired |
| NumPy BeaconCNN | trained (beacon-numpy-v1) | EXPERIMENTAL, wired with fallback |
| Disturbance MLPs ×3 | trained | UNUSED |
| Failure predictor | trained artifact / heuristic at runtime | NOT LOADED |

No model claims superiority over its baseline without the measured
number beside it. The two head-to-heads both favor the baseline:
CV 4.51 px over GRU 35.16 px; classical detector over visual CNN.
