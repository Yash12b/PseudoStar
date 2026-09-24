# Technical Report — AI-Based Virtual Camera Tracking System for Coarse Alignment of Mobile FSOC Terminals

## 1. Problem understanding

Free-space optical communication (FSOC) offers gigabit-to-terabit data
rates over license-free spectrum with high immunity to electromagnetic
interference, which makes it attractive for next-generation mobile
networks spanning satellites, UAVs, and ground vehicles. The same
directionality that enables this performance is also the core
difficulty: the optical beam is extremely narrow, so even a small
angular pointing error breaks the link. Establishing a link therefore
requires a pointing, acquisition, and tracking (PAT) sequence, executed
in two stages — coarse alignment followed by fine alignment.

Coarse alignment is the stage this project addresses. Before any fine
pointing mechanism can take over, the transmitting terminal must first
locate the remote terminal (or its beacon) within the camera
field-of-view, estimate its position and motion, and continuously steer
the pointing direction so the target stays visible. Concretely, the
coarse-alignment loop must: observe the surrounding environment,
acquire and detect the remote beacon, estimate its position (centroid),
and continuously adjust the pointing direction to maintain visibility —
including recovery when the target is temporarily lost.

Developing and validating such algorithms on real hardware demands
expensive cameras, pan-tilt mechanisms, and optical equipment. A
software virtual camera tracking system provides an inexpensive,
accessible platform for algorithm development and learning, which is
exactly what the problem statement (SIH26169, Department of Space /
ISRO) calls for: a software system that autonomously detects,
identifies, and continuously tracks a designated moving target within a
virtual scene by controlling a virtual camera viewport.

The reference operating point used throughout this project is taken
directly from the problem statement: a 640×480 monochrome camera with
a 4°×3° field of view updating at 30 Hz, mounted on a virtual pan-tilt
stage limited to 5°/s in both axes, observing beacon-spot targets of
5–20 px (default 10×10) moving on selectable trajectories (straight
line, circular, figure-of-8, random as mandatory; spiral, sinusoidal,
and user-defined as options). The performance targets are acquisition
≤ 2 s, tracking error ≤ 10 px, target loss < 5%, re-acquisition ≤ 1 s,
and processing ≥ 20 FPS. Disturbances — salt-and-pepper, Gaussian, and
Poisson image noise, camera jitter up to ±20 px/frame, atmospheric
effects (clear, haze, fog, rain, low light), and platform motion up to
±20 px/frame on selectable paths — must be generatable and injectable
into the virtual camera feed. Evaluation is multi-layered: functional
demonstration (20%), benchmark performance on given scenarios (30%),
benchmark performance on supplied MP4 video files at 30 fps with the
PTZ camera bypassed (30%), and technical presentation (20%). Every
design decision in this system traces back to one of these clauses; a
full clause-by-clause mapping is maintained in
`docs/requirements/sih26169-traceability.md` and summarized in §9.

## 2. System architecture

The system is organized as a single closed loop with a strict
information boundary:

```
SIMULATION / MP4 VIDEO / LIVE CAMERA
        │ (FrameSource → Frame: image, timestamp, dims, metadata)
        ▼
PREPROCESSING (noise-gated median denoise, background estimation)
        ▼
PERCEPTION → BEACON DETECTION → CENTROIDING (intensity-weighted)
        ▼
TRACKER (Kalman, constant-velocity motion, dt from timestamps)
        ▼
PREDICTION (constant-velocity production; GRU experimental)
        ▼
SITUATION / FAILURE RISK → POLICY/BRAIN → SAFETY → PAN/TILT CONTROL
        ▼
CAMERA / VIEWPORT ──→ NEXT FRAME (closed loop)
```

Three interchangeable observation sources — a configurable simulation,
an external MP4 video file, and a live camera — all feed one shared
tracking pipeline (`pipeline/pipeline.py`, `TrackingPipeline`). The
pipeline is deliberately source-agnostic: it consumes only `Frame`
objects (image pixels, timestamps, dimensions) and never branches on
source type for perception, tracking, or control logic. This is what
makes the Benchmark Performance-2 requirement (MP4 input bypassing the
PTZ camera) a configuration choice rather than a separate code path.

The information boundary is absolute and enforced by tests: world
ground truth flows only into rendering, offline label generation, and
benchmark scoring. Runtime perception, tracking, AI, and control
receive image pixels, timestamps, and own-history state only. Frame
metadata carries no ground truth in any production path. A dedicated
`EvalSink` side channel attaches ground truth to simulation sources for
offline scoring without ever exposing it to the runtime loop.

Concretely, leakage control has four layers. First, the frame contract:
`Frame` carries image, timestamp, dimensions, and nominal fps — the
constructor raises if ground truth appears in metadata. Second, the
source contract: `VirtualSimulationSource` never attaches truth to
frames; the paired `EvalSink` is keyed by frame index and readable only
by scoring code. Third, static guards: tests fail the build if
simulation modules are imported from runtime (perception, tracking,
control, AI) paths. Fourth, dynamic guards: the MP4 acceptance suite
includes a guarded-annotation test whose runtime raises on any
ground-truth access. Determinism is verified the same way — seeded
repeat runs must produce identical trajectories, detections, and
metrics — so every reported number is reproducible from its recorded
repro command.

The closed loop meets its timing budget with wide margin: the nominal
30 Hz cadence allows 33.3 ms per frame, while measured pipeline
latency is ~3 ms (309–361 FPS headless, 312 FPS live in the GUI with
full overlay composition), leaving the budget free for disturbance
rendering and AI inference on every frame.

The operator sees the same loop the pipeline executes. The camera
tracking page renders the live frame with a square-bracket reticle
(+ centroid marker) in a state-encoded color — green locked, orange
tracking/acquiring, yellow detected-but-unlocked, red lost — plus the AI
prediction diamond with horizon tag, dotted uncertainty circles, camera
crosshair, and state/mode text, all baked into the displayed pixmap so
overlays can never be obscured by the video layer. Below the frame, a
telemetry grid reports target position, centroid, per-axis and Euclidean
error, confidence, velocity, prediction and uncertainty status,
pan/tilt, FPS, and latency every refresh; the analysis page adds error
plots, the event log records state transitions, and the scorecard
accumulates acquisition time, RMSE, loss percentage, and reacquisition
statistics for the run.

Twenty absolute engineering rules govern development (no hidden ground
truth at runtime, no fabricated results, no test-set training, no
validation leakage, no future-information leakage, deterministic
fallback always available, safety dominates AI, existing functionality
preserved, no deleting tests to get green builds, measure-before-claim,
and never hardcode frame rate — dt always comes from timestamps). These
rules are referenced in code review and in the AI audit
(`docs/ai/final-audit.md`).

### 1.2 How the system serves each evaluation stage

Functional verification (20%) is served by the desktop application
itself: a 10–15 minute demonstration can create a world, spawn beacons,
start closed-loop tracking, inject disturbances live, load an MP4, and
show the auto-generated performance report — all mandatory functions
operable from the GUI. Benchmark Performance-1 (30%) is served by the
benchmark harness (`benchmark/runner.py`, CLI `fsoc-bench`): named
world profiles execute headlessly and emit centroid-error logs plus
JSON/CSV performance logs with repro commands. Benchmark
Performance-2 (30%) is served by VIDEO mode: the pipeline bypasses the
virtual PTZ camera, consumes the supplied 30 fps MP4 through
`VideoSource` with decoder timestamps, and reports RMSE, acquisition
and re-acquisition times, lock retention, and FPS against the same
scoring code as simulation. Technical evaluation (20%) is served by
this report, the architecture/algorithms documentation set, the
12-model AI audit, and the traceability matrix.

### 1.1 Reference parameters (problem statement operating point)

| # | Parameter | Value in this system |
|---|-----------|----------------------|
| 1 | Screen (world) size | 2000×2000×2000 units, user-configurable |
| 2 | Camera type | Monochrome focal-plane array (colour-capable pipeline) |
| 3 | Camera resolution | 640×480, user-configurable |
| 4 | Camera FOV | User-defined, default 4°×3° |
| 5 | Camera update rate | 30 Hz |
| 6 | Initial camera position | Screen centre (1000, 1000, 50) |
| 7–8 | Target type / count | Beacon spot; 1 mandatory, multiple supported |
| 9–10 | Target shape / size | User-defined (square default); 5–20 px, default 10×10 |
| 11 | Initial target location | User-defined or random (seeded) |
| 12 | Motion | Straight line, circular, figure-8, random (mandatory) plus spiral, sinusoidal, random-walk, stop-go, sudden-reversal, accel-decel, user-controlled |
| 13–14 | Max pan / tilt speed | 5°/s default, user-adjustable 5–10°/s |
| 15 | Control update interval | ≥ 20 Hz (30 Hz nominal) |
| 16–20 | Performance targets | Acquisition ≤ 2 s; error ≤ 10 px; loss < 5%; re-acquisition ≤ 1 s; ≥ 20 FPS |
| 21 | Disturbances | Selectable salt-and-pepper/Gaussian/Poisson noise (σ ≤ 20), jitter (±20 px/frame), haze/fog/rain/low-light atmosphere, linear/circular/figure-8/spiral/random platform motion (±20 px/frame) |

## 3. Software modules

- **Simulation** (`simulation/`): deterministic engine advancing world
  time and evaluating trajectories; a generic world builder
  (`world_builder.py`) replacing the retired scene presets, with
  benchmark world profiles (nominal, multi-beacon, distractor, loss,
  noise, fog, jitter, fast); 11 trajectory types (straight line,
  circular, figure-8, random, random-walk, spiral, sinusoidal, stop-go,
  sudden-reversal, accel-decel, user-controlled); a sensor renderer
  depositing square, circular, or soft-Gaussian beacon spots
  (per-target shape, 5–20 px clamped apparent size) with a point-spread
  function; an authoritative virtual PTZ camera model (resolution, FOV,
  pose, intrinsics, 3D-to-2D projection); plus optical-link and
  communication engines for the link layer.
- **Frame sources** (`pipeline/sources.py`): `VirtualSimulationSource`
  (simulation with EvalSink side channel), `VideoSource` (decoder-first
  timestamps, arbitrary dimensions/rates, seek support, equirectangular
  auto-detection with viewport extraction), `LiveSource` (camera
  availability probing with graceful errors), `LiveViewportSource`
  (digital PTZ crop steering when no hardware PTZ exists).
- **Perception** (`perception/`): classical bright-spot detector
  (preprocess → percentile-threshold candidates → feature extraction →
  intensity-weighted centroid with NaN/Inf and zero-weight fallbacks),
  two-stage fallback thresholds (80th then 50th percentile) for dim
  video beacons, a `for_video()` configuration (relaxed percentile,
  contrast normalization, widened size tolerance) auto-applied to
  video/live sources, adaptive ROI with FULL_FRAME fallback and
  risk-preemptive widening, and weighted candidate scoring
  (intensity/size/shape/contrast).
- **Tracking** (`tracking/`): 2D Kalman filter with Joseph-form update
  and residual/Mahalanobis gating, an adaptive-Kalman variant, a
  track-state machine
  (NO_TRACK→SEARCHING→ACQUIRING→TRACKING→LOST→REACQUIRING), a search
  controller (last-known local search, uncertainty region, predictive
  search, expanding spiral, global sweep) with origin-relative waypoints
  that never parks the camera, nearest-neighbor plus Mahalanobis data
  association, lock-quality scoring, and maneuver detection. Timestamps
  — never a hardcoded frame rate — drive all dt computations.
- **Control** (`control/`): PID controller with deadband, integral
  clamp, filtered derivative, output saturation and anti-windup, and
  NaN/Inf guards; PS-spec default gains (kp=0.8, ki=0.02, kd=0.08);
  opt-in bounded lead-angle compensation gated by the AI brain to fast,
  high-confidence, safety-approved motion only; a rate-limited actuator
  applying commands to the virtual mount.
- **AI** (`ai/`): a rule-based situation classifier (10 categories:
  normal, fast motion, high noise, edge-of-FOV, degrading track, low
  confidence, prediction uncertainty, lost, reacquisition, recovery), an
  expert mission policy (production), a ridge-multinomial learned policy
  (behavioral clone that loads only after exact 11-label validation), a
  motion predictor with configurable horizon, a ridge failure-risk
  predictor running on trained weights over sliding observation
  windows, and a safety envelope plus safety gate that dominates every
  AI decision. Twelve models are tracked with per-model provenance
  labels (trained / random / imported / deterministic_baseline); the
  temporal GRU, visual CNN, and disturbance MLPs are trained but
  quarantined by measured evidence because the classical baseline beat
  them on the audited protocol.
- **Disturbances** (`disturbances/`): 14 effect types across image,
  camera, and source layers — Gaussian, salt-and-pepper (~10%), and
  Poisson noise individually selectable with user-defined sigma (PS max
  20); camera jitter up to ±20 px/frame on bounded/Gaussian/sinusoidal/
  damped models with user-defined amplitude; atmospheric haze, fog,
  rain, and low light with contrast/brightness reduction; platform
  motion up to ±20 px/frame on linear (default), circular, figure-8,
  spiral, and random paths; plus blur, motion blur, brightness/contrast,
  turbulence, distractors, and temporal target disappearance with real
  configured durations. Severity presets (clear/light/moderate/severe/
  extreme) combine with per-effect numeric tuning in the GUI.
- **GUI** (`gui/`): a PySide6 desktop application with five navigation
  pages (mission setup, camera tracking, analysis, world, benchmark),
  a camera-first workspace rendering the live frame with a
  square-bracket tracking reticle (+ centroid marker), AI prediction
  diamond, uncertainty circles, and telemetry; a 3D setup world view; a
  benchmark panel; a mission flow strip; five layout presets; floating
  AI/link/telemetry/event panels; and a visible LOAD VIDEO action
  beside the source selector.
- **Benchmark** (`benchmark/`, `cli/`): five benchmark methods with
  honest NOT AVAILABLE gating, per-scenario JSON+CSV logs, multi-seed
  mean/std aggregation, repro commands, and an automatically generated
  performance report (duration, FPS, acquisition time, average/maximum
  error, lock retention, processing time).

### 3.1 Key classes and entry points

| Component | Key classes / functions |
|---|---|
| World & engine | `WorldConfig`, `SimulationEngine`, `world_builder.add_beacon`, `TrajectoryRegistry` |
| Camera & sensor | `VirtualCamera`, `CameraState`, `CameraIntrinsics`, `VirtualSensorRenderer`, `deposit_beacon` |
| Sources | `VirtualSimulationSource`, `VideoSource`, `LiveSource`, `LiveViewportSource`, `EvalSink` |
| Perception | `ClassicalBeaconDetector`, `PerceptionConfig` (+`for_video()`), `AdaptiveROI`, `score_candidate` |
| Tracking | `KalmanTracker`, `TrackingState`/`TrackState`, `SearchController`, association gates |
| Control | `CoarsePointingController`, `CameraActuator`, lead-angle compensation |
| AI | `AIMissionBrain`, `MissionAction`, `Situation`, failure predictor, safety gate |
| Disturbances | `DisturbancePipeline`, `NoiseConfig`, `JitterConfig`, `AtmosphereConfig`, `PlatformMotionConfig` |
| GUI | `MainWindow`, `CameraTrackingWorkspace`, `ControlPanel`, `ProcessingWorker`, `TrackingPipeline` |
| Benchmark | `benchmark/runner.py`, `ai_benchmark.py`, `report.generate_performance_report` |
| CLI | `fsoc-sim` (`cli/run_simulation.py`), `fsoc-bench` (`cli/run_benchmark.py`) |

## 4. Tracking methods

Production estimation is classical computer vision, selected by
measurement rather than fashion. Detection thresholds a preprocessed
grayscale frame at a high brightness percentile, extracts connected
components in the 5–20 px size band, scores candidates on intensity,
size, shape, and local contrast, and centroids the winner with
intensity weighting (0.19 px mean error on the noisy 30 fps reference
video). The Kalman tracker runs a constant-velocity motion model with
Joseph-form covariance updates, enforced symmetry after predict and
update, and Mahalanobis gating for association;
a raw-detection (no-filter) method is benchmarked alongside purely to
prove the filter's value (63% target loss without it versus 1.5% with
it on the nominal scene). A BYTE-style second-chance pass revives live
tracks with sub-threshold detections near prediction (capped streak,
never births tracks), lifting lock retention from 51% to 90% on
flickering-dim content. Association additionally supports an opt-in
appearance term (running size/brightness signature learned only from
lock-quality frames); in 30 seeded planted-glint trials after a sharp
maneuver it cuts hijacks from 24/30 (geometry only) to 0/30 —
measured by `tests/unit/test_association_appearance.py`, which the
benchmark suite runs. On the adversarial 6-lookalike distractor world
the term does not move aggregate RMSE (6.63 px with weight 0 and
weight 8 alike — geometry gaps dominate there); that world still
needs true multi-hypothesis tracking (birth luck dominates). A
second opt-in association aid is coded beacon identity (temporal
on/off keying): the true beacon blinks an operator-configured binary
code, decoys blink the inverse or burn steady, and the tracker chains
per-candidate intensity histories, decodes them against the expected
code, and skips confidently mismatching candidates before ranking —
coasting through the true beacon's OFF gaps instead of hijacking. On
the adversarial `coded` world (primary 10110010 vs five equal
decoys, code the ONLY separator; kalman_expert, seeds 42/7/13, 600f),
identity at weight 40 cuts RMSE 12.08→2.16 px and false-lock frames
54.8%→2.5% (72→2 events), measured with an on/off ablation. Lock
retention reads lower (99.5%→51.5%) because a 50%-duty beacon yields
measurable photons half the time — reported alongside, not hidden.
Covered by `tests/unit/test_identity.py` (19 tests). The
benchmark harness feeds the tracker the full above-threshold
detection list (not just the primary), so association and the
appearance term face real competition. LAST_POSITION,
constant-velocity, and Kalman predictors are preserved in the
evaluation code for comparison. The
benchmark suite additionally reports precision/recall and MOTA/MOTP/IDF1.

Acquisition works by global-frame detection or, after loss, by the
search controller's escalating strategies; re-acquisition replays the
same chain. An audited comparison on a thin protocol gave the Kalman
baseline 4.51 px RMSE against 35.16 px for the experimental GRU — the
baseline wins and stays in production while the GRU remains an
experiment. Recovery behavior (search → re-acquire → resume track with
ROI reset) is covered by a dedicated 33-test recovery suite plus the
end-to-end regression.

Two ideas were adopted from a review of a competing SIH26169
implementation (Team Astrionics video + repository): the expanding
spiral now steps at constant linear speed along the path
(dθ = S/√(b²+r²), verified uniform 12 px/frame) instead of fixed
angular steps whose outer loops race, and the 3D world renders the
camera sweep trail plus target motion trails. Their morphological
top-hat prefilter was prototyped and measured against our
percentile-plus-contrast-normalization chain on fog, haze, and
low-light footage: ours detects 60/60 at ~0.06 px centroid error with
no prefilter, and top-hat scored worse under extreme fog+noise (it
amplifies bright noise specks past the beacon), so it was not adopted —
the measurement, not the idea, decided.

### 4.1 Detection chain detail

Each frame is converted to grayscale and background-estimated (default
10th percentile, 5th for video sources). Candidate regions are pixels
above `background + dynamic_range × percentile` (95th by default, 75th
for video/live via `PerceptionConfig.for_video()`, with automatic
retries at the 80th and 50th percentiles when the strict pass is
empty), filtered to the configured size band and scored as a weighted
sum: 0.30 intensity + 0.25 size + 0.25 shape + 0.20 contrast
(0.25/0.20/0.25/0.30 for video, where local contrast discriminates
better). The winner above `min_confidence` (0.3 default, 0.2 video) is
centroided with intensity weighting; geometric-centroid, NaN/Inf, and
zero-weight fallbacks guarantee a finite answer or an explicit
NO_TARGET status — never a silent NaN.

### 4.2 Tracker and search detail

The Kalman filter tracks 2D position/velocity with a Joseph-form
covariance update for numerical stability. Association gates detections
by residual and Mahalanobis distance (configurable, with a Euclidean
floor to avoid overconfident rejection). The state machine advances on
consecutive detections/misses with configured thresholds; lock requires
the TRACKING state plus a quality score (confidence, uncertainty, and
streak factors) above 0.5 with zero outstanding misses. The search
controller issues origin-relative waypoints on an expanding repertoire
— last-known local search, uncertainty-region search, predictive
search along the velocity vector, expanding spiral, global sweep — and
recycles rather than parking the camera when a pattern expires, so a
target drifting back into view is always catchable. Adaptive ROI
follows the estimate (stable/shrinking when confident, expanding when
uncertain, full-frame on loss) and widens preemptively on high failure
risk. Optional lead-angle compensation extrapolates the rate command
along the estimated velocity with a bounded lookahead, and only when
the mission brain explicitly commands predictive tracking for fast,
high-confidence, safety-approved motion.

## 5. AI methods

AI assists — never overrides — the classical loop. A deterministic
rule-based situation classifier maps observable features (detection
confidence, miss streak, edge proximity, noise estimates, prediction
uncertainty) to 10 situation categories; an expert policy maps
situations to 11 mission actions (track, predictive track, ROI modes,
classical/hybrid perception, local/global search, reacquire, hold, safe
stop). A learned ridge-multinomial policy distills the expert (≈98.9%
mimicry accuracy — reported strictly as a distillation score, not an
outcome claim) and is admitted at runtime only after exact 11-label
validation. Failure risk comes from trained ridge weights over sliding
observation windows and drives preemptive ROI widening before misses
accumulate. Motion prediction uses the classical constant-velocity
extrapolator in production with a configurable horizon; the temporal
GRU predictor is trained, measured, and quarantined. Every AI decision
passes a safety gate, and safety approvals dominate: lead-angle
compensation, for example, engages only for explicitly
brain-commanded predictive tracking of fast targets with confidence
above 0.5 and safety approval. The full 12-model provenance map,
training protocols, and quarantine rationale live in
`docs/ai/final-audit.md`.

A trained neural detector now exists alongside the classical one: a
15K-parameter BeaconCNN trained in PyTorch on 4000 synthetic samples
and exported weight-for-weight to the NumPy runtime
(`artifacts/models/beacon-numpy-v1`, parity 6e-06, 40/40 training
sanity). It is selectable live as the `ai` backend (or fused in
`hybrid`), with missing weights falling back to classical under a
logged warning — never silent random weights. Measured honestly on
the noisy 640×480 reference video it fires on noise structure
(RMSE 297 px vs 0.32 px classical; hybrid 230 px), so it ships as an
explicitly experimental ablation vehicle, not a replacement.
Classical remains the production default.

## 6. Test methodology

The suite holds 1316 passing tests (2 skipped): unit tests per module
(perception, tracking, control, disturbances, world builder, camera,
sensor, trajectories, AI safety, GUI wiring, recovery chain, pipeline
chain, closed loop), integration tests (end-to-end pipeline, external
video acceptance including the PS 30 fps noisy case, generic world
workflow), and a flagship end-to-end regression exercising multi-beacon
setup, misaligned start, outage blackout, search/ROI behavior,
reacquisition, and link restore — all image-driven, never
ground-truth-driven. Determinism checks assert repeat runs are
identical; static guards forbid simulation imports in runtime paths; a
guarded-annotation MP4 test raises on any runtime ground-truth access.
Video-mode detection, overlay composition, shape rendering, and
disturbance tuning each carry dedicated regression tests. The same
methodology governs the packaged executable, which must pass the smoke
test standalone before release.

### 6.1 Suite composition

| Area | Coverage |
|---|---|
| Perception | Detection accuracy, centroid error, size robustness (5–20 px), noise robustness, determinism, video-mode config |
| Tracking | Kalman convergence, association (Euclidean + Mahalanobis), state-machine transitions, search strategies, recovery chain (33 tests) |
| Control | PID step response, saturation/anti-windup, rate limits, lead compensation gating |
| Disturbances | Per-effect rendering, presets, temporal disappearance windows, GUI tuning overrides |
| Simulation | Engine determinism, trajectories, camera projection, sensor rendering, shape wiring |
| AI | Situation/policy determinism, label validation, safety gating, provenance |
| GUI | Window wiring, mission setup, video source selection, overlay composition, config plumbing (150+ tests) |
| Integration | End-to-end closed loop, external MP4 acceptance (14 tests incl. 30 fps noisy reference), world workflow |

## 7. Performance analysis

Measured results on the reference configuration (kalman_expert policy,
seed 42, 200-frame runs): nominal scene — acquisition 0.10 s, RMSE
0.19 px, loss 1.5%, 321 FPS processing; temporal-outage scene — 3 loss
events with re-acquisition 0.067 s from GT reappearance (1.07 s raw,
including the outage window) and 0.89 px RMSE; multi-beacon
scene — RMSE 12.38 px (above the 10 px target, openly reported);
distractor field with glints — RMSE 6.63 px with 33% loss (45 events;
the full-detection-list harness cut this from 118.84 px / 72% loss,
but identity across 6 lookalikes remains an association limit);
coded-identity duel (seed 42, 200f) — RMSE 10.81 px geometry-only vs
4.25 px with identity at weight 40 (code 10110010), acquisition
0.10 s both; over 600f × 3 seeds false-lock frames fall 54.8%→2.5%;
two moving endpoints (seed 42, 200f, platform drift 0.3 u/s + yaw
0.2°/s) — acquisition 0.10 s, RMSE 0.40 px, lock 98.5%, 327 FPS
(600f × 3 seeds: 0.26 px / 99.5%);
full-AI mission on the nominal scene — RMSE 0.17 px, 1.0% loss,
1975 FPS. On the MP4 640×480@30 fps noisy reference: 100%
detection, 0.37 px centroid error, 361 FPS. GUI live runs sustain
312 FPS with the link LOCKED at 0.14°. Against the PS targets:
acquisition ≤ 2 s ✓, tracking error ≤ 10 px ✓ (nominal), loss < 5% ✓
(nominal), re-acquisition ≤ 1 s ✓ (0.067 s from reappearance),
processing ≥ 20 FPS ✓ by an order of magnitude. Stress scenes exceed
the error/loss targets and are reported as such — no universal claim
is made. Every number ships with scenario, method, seed, and frame
count attached, and the benchmark harness regenerates them via
recorded repro commands.

### 7.1 Results vs problem-statement targets

| PS target | Measured (nominal) | Verdict |
|---|---|---|
| Acquisition ≤ 2 s | 0.10 s | ✓ met |
| Tracking error ≤ 10 px | 0.19 px RMSE (0.37 px centroid on MP4) | ✓ met |
| Target loss < 5% | 1.5% (1.0% full-AI) | ✓ met |
| Re-acquisition ≤ 1 s | 0.067 s from GT reappearance (outage scene) | ✓ met |
| Processing ≥ 20 FPS | 171–511 FPS benchmark; 312 FPS GUI live | ✓ met by an order of magnitude |

Stress scenes (multi-beacon 12.38 px RMSE; distractor field 33% loss
across 6 lookalikes; coded duel 10.81 px geometry-only / 4.25 px with
identity) exceed the error/loss targets and are reported
as association limits, not averaged away. GUI performance reports
store the full raw error series beside the JSON and self-verify on
every run (`run_benchmark verify` recomputes all aggregates from raw;
tampered numbers fail). A GT-free HANDOFF READY signal fires after 15
consecutive locked low-uncertainty frames, marking coarse stability
for a future fine-pointing stage.

## 8. Limitations and future improvements

Equal-brightness multi-target identity can still hand the track to a
lookalike on birth (the distractor world loses 33% of frames across
6 identical decoys); deeper appearance features or joint probabilistic
association would address it. The opt-in size/brightness signature
already defeats the planted near-glint case (24/30 → 0/30 hijacks),
and coded identity defeats equal-brightness decoys when the beacon
transmits a code (coded world false-locks 54.8%→2.5%). Known identity
limit: beacons crossing within ~15 px merge chains and contaminate
histories for up to ~4 code periods — the failure mode is safe
(coast, don't hijack) but costs retention during prolonged crossings.
Terminal A itself can now move (platform drift + attitude rates,
strapdown mount carry); the two-moving-endpoints world holds
0.26 px / 99.5% over 600f × 3 seeds. Platform motion is
constant-velocity only (no orbital dynamics), and gimbal travel
limits remain unmodeled.
The virtual pan stage has rate limits but no range stops. Benchmark
runs use single seeds unless `--seeds` is passed. Live-camera hardware
remains unvalidated beyond graceful open-failure handling. The
quarantined GRU/visual models need larger, more diverse datasets to
challenge the classical baseline honestly. The five GUI layouts share
page widgets (combinations, not bespoke compositions). Fine-pointing
handoff is out of scope — this system is coarse alignment only, by
design (though it now emits a HANDOFF READY stability signal for a
future fine stage). Near-term improvements: richer per-target
appearance features, range-stop modeling, seed-swept benchmark
defaults, and a hardware validation pass on a real camera + pan-tilt
stage.

### 8.1 Future work per limitation

- **Association robustness.** The distractor-field 33% loss (6
  identical decoys) is the system's worst measured behavior and
  therefore its most important research direction: richer per-target
  appearance features (spot-shape histograms, temporal flicker
  signatures) and joint probabilistic data association would attack
  the birth-identity failure mode the current size/brightness term
  cannot reach, with the existing distractor world profile as the
  acceptance test. Coded identity already solves the sub-case where
  the beacon transmits a temporal code (coded world, 54.8%→2.5%
  false-lock frames); the remaining gap is unkeyed lookalikes.
- **Mount modeling.** Adding configurable pan/tilt range stops and
  slew-rate profiles would let missions validate edge-of-travel
  behavior and keep the search controller honest near limits.
- **Benchmark rigor.** Making multi-seed sweeps the default (rather
  than opt-in `--seeds`) would turn every reported number into a
  mean±std by construction.
- **Hardware validation.** A validation pass on a real monochrome
  camera and pan-tilt stage — reusing the live-source path already
  exercised with graceful open-failure handling — would close the
  sim-to-real gap the virtual laboratory is designed to minimize.
- **Learned models.** Larger, more diverse labeled datasets are the
  precondition for re-admitting the quarantined GRU/CNN models; the
  audit protocol (thin-comparison, provenance labels, quarantine on
  loss) already exists to keep that process honest.
- **Fine-pointing handoff.** The coarse loop currently terminates in a
  stable track with quantified uncertainty — the natural handoff
  record for a future fine-alignment stage, which would consume the
  estimate, covariance, and lock-quality signals defined here.

## 9. Problem-statement traceability summary

Camera: 2000×2000 minimum world with user-defined sizing; monochrome
640×480 sensor (colour-capable pipeline); user-defined FOV defaulting
to 4°×3°; 30 Hz update; centre-screen initial pose. Targets: beacon
spots, single mandatory with multi-beacon option; user-defined shape
defaulting to square; 5–20 px user-defined size defaulting to 10×10;
user-defined or random initial placement; straight-line, circular,
figure-8, and random motion plus spiral, sinusoidal, random-walk, and
user-controlled options. Motion constraints: 5°/s default pan/tilt caps
(user-adjustable 5–10°/s), ≥20 Hz control updates. Performance:
acquisition, error, loss, re-acquisition, and FPS targets instrumented
in the benchmark suite and auto-reported. Disturbances: selectable
salt-and-pepper/Gaussian/Poisson noise (σ ≤ 20), jitter (±20 px/frame),
haze/fog/rain/low-light atmosphere, and linear/circular/figure-8/
spiral/random platform motion (±20 px/frame). Deliverables: standalone
executable, documented modular source, this report, a user manual with
installation/operation/configuration/GUI documentation, a demonstration
video, and auto-generated performance logs. The clause-by-clause matrix
is maintained in `docs/requirements/sih26169-traceability.md`.

### 9.1 Deliverables checklist

| Deliverable | Location / form |
|---|---|
| Standalone executable | `dist/FSOC-Tracker` (PyInstaller bundle, smoke-tested standalone); run-from-source fallback `python -m fsoc_tracker` |
| Source code | `src/fsoc_tracker/` (~18 packages), modular with docstrings; `requirements.txt` + `requirements-dev.txt` pin the environment |
| Technical report | This document (10–15 pages) |
| User manual | `docs/user/manual.md` — installation, operation, parameter configuration, GUI description, troubleshooting, command reference |
| Demonstration video | `artifacts/demo/` — closed-loop tracking demonstration (see manual §10) |
| Performance log | Auto-generated per run: `artifacts/reports/report_<ts>.json` + `.txt` summary (duration, FPS, acquisition, avg/max error, lock retention, processing time) |
