# FSOC Tracker — User Manual

**Version:** 0.1.0
**SIH Problem Statement:** SIH26169

---

## 1. Installation

### Prerequisites
- Python 3.10 or later
- pip

### Install from source

```bash
git clone <repository-url>
cd fsoc_tracker

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install with all dependencies
pip install -e ".[dev,full]"

# Verify installation
python -m fsoc_tracker
```

### Optional dependencies

```bash
# AI perception (requires torch)
pip install -e ".[ai]"

# Full install (dev + AI + plotting)
pip install -e ".[dev,ai,full]"
```

---

## 2. Quick Start

### Smoke test (real pipeline, 5 seconds)

```bash
python -m fsoc_tracker
```

### Headless simulation

```bash
# Basic simulation (straight line, 10 seconds)
fsoc-sim

# Circular trajectory with disturbances
fsoc-sim -t circular -d 10 --disturbance light

# All options
fsoc-sim --trajectory figure_8 --duration 20 --dt 0.033 --seed 42 \
         --target-size 10 --disturbance moderate --output runs/ --verbose
```

### Video benchmark

```bash
# Benchmark an MP4 file
fsoc-bench -i video.mp4

# With verbose output
fsoc-bench -i video.mp4 -v --output runs/benchmark
```

### Benchmark suites (generated worlds + seeds)

```bash
# Single generated-world run
python -m fsoc_tracker.cli.run_benchmark sim --method kalman_expert \
    --world multi --seed 42 --frames 300 --output logs/benchmark

# Multi-seed aggregation (mean/std across seeds)
python -m fsoc_tracker.cli.run_benchmark sim --method kalman_expert \
    --world loss --seeds 42,43,44 --frames 300 --output logs/benchmark

# Opt-in appearance-gated association (anti-hijack; default 0 = off).
# Measured: planted-glint hijacks 24/30 -> 0/30 at weight 8.
python -m fsoc_tracker.cli.run_benchmark sim --method kalman_expert \
    --world distractor --seed 42 --frames 300 --assoc-appearance 8
```

### GUI

```bash
python -m fsoc_tracker --gui
```

---

## 3. Configuration

All parameters are configurable via YAML files in `configs/`.

### Configuration profiles

| Profile | File | Description |
|---------|------|-------------|
| Default | `configs/default.yaml` | Standard SIH26169 defaults |
| Production | `configs/production.yaml` | All advanced features enabled |
| Baseline | `configs/baseline.yaml` | Minimal (no advanced features) |
| Debug | `configs/debug.yaml` | Full diagnostics + logging |
| Benchmark | `configs/benchmark.yaml` | Video benchmark settings |

### Using a config file

```bash
python -m fsoc_tracker -c configs/production.yaml
fsoc-sim -c configs/debug.yaml -t circular
```

### Key parameters

```yaml
camera:
  width: 640              # Image width (pixels)
  height: 480             # Image height (pixels)
  horizontal_fov_deg: 4.0 # Horizontal FOV (degrees)
  vertical_fov_deg: 3.0   # Vertical FOV (degrees)
  update_rate_hz: 30.0    # Camera update rate (Hz)

target:
  size_px: 10.0           # Beacon size 5-20 px (PS default 10x10)
  shape: square           # Beacon shape: spot (soft, default rendering),
                          # square or circular (hard-edged spots)
  motion_type: straight_line  # straight_line | circular | figure_8 |
                              # random | spiral | sinusoidal | user_controlled

control:
  max_pan_speed_deg_s: 5.0   # Max pan rate (deg/s)
  max_tilt_speed_deg_s: 5.0  # Max tilt rate (deg/s)
  pid:
    kp_pan: 0.5              # Pan proportional gain
    ki_pan: 0.01             # Pan integral gain
    kd_pan: 0.1              # Pan derivative gain

disturbance:
  enabled: false             # Enable disturbances
  types: []                  # List of disturbance types
  gaussian_sigma: 5.0        # Gaussian noise sigma (PS max 20)
  salt_pepper_density: 0.05  # S&P density (PS ~10% => 0.10)
  poisson: true              # Photon-shot Poisson noise on/off
  jitter_amplitude_px: 5.0   # Camera jitter (PS max ±20 px/frame)
  platform_type: linear      # linear | circular | figure_eight |
                             # spiral | random (PS default linear)
  platform_amplitude_px: 10.0  # Platform motion (PS max ±20 px/frame)
```

In the GUI disturbance tab the same controls are widgets: master
effect checkboxes, Gaussian / Salt&Pepper / Poisson subtype switches
(all on = legacy behavior), Noise σ (0–20), S&P density (0–0.5),
Jitter amp (0–20 px), and a Platform path combo. Presets
(clear/light/moderate/severe/extreme) set the master switches and
leave numeric tuning to you; changes apply live to a running worker.

---

## 4. Operation Modes

### Simulation Mode

The full closed-loop simulation:
1. Virtual world generates target position
2. Virtual camera renders the scene
3. Sensor creates beacon image
4. Disturbances are applied (optional)
5. Perception detects the beacon
6. Tracker estimates position and velocity
7. Controller computes pan/tilt commands
8. Camera moves → loop

```bash
fsoc-sim -t circular -d 30
```

### Video Mode

Process an external video file:
1. Video decoder reads frames
2. Perception detects beacon in each frame
3. Tracker estimates position
4. Metrics are computed against ground truth (if available)

```bash
fsoc-bench -i benchmark_video.mp4 -v
```

### Live Camera Mode

Real-time webcam processing:
1. Camera captures frames
2. Perception + tracking in real-time
3. Control commands displayed (not sent to physical camera)

```bash
python -c "
from fsoc_tracker.pipeline.sources import LiveSource
from fsoc_tracker.pipeline.pipeline import TrackingPipeline
from fsoc_tracker.perception.classical_engine import ClassicalBeaconDetector
from fsoc_tracker.tracking.tracker import KalmanTracker

source = LiveSource(camera_id=0)
pipeline = TrackingPipeline(
    perception=ClassicalBeaconDetector(),
    tracker=KalmanTracker(),
)
pipeline.set_source(source)
pipeline.start()
while pipeline.running:
    result = pipeline.step()
    if result:
        print(f'Frame {result.frame_index}: {result.tracking.state.name}')
pipeline.stop()
"
```

---

## 5. GUI Overview

The GUI provides a real-time aerospace HUD with:

- **Camera View** — Live camera feed with detection overlay
- **World View** — 2000x2000 global radar with trajectory trail
- **Telemetry** — Real-time tracking and control metrics
- **Scorecard** — SIH threshold evaluation
- **Error Plots** — Live X/Y/Euclidean error graphs
- **Event Log** — Color-coded tracking events
- **Control Panel** — Configuration and mode switching

### Launch

```bash
python -m fsoc_tracker --gui
```

### World workflow (simulation mode)

There are no preset scenes. Every simulation run starts from an
explicitly created world:

1. Top bar, WORLD menu: **New empty world** or **Random world (seed)**
   (uses the seed spin box; same seed reproduces the same world).
2. Place Terminal A (3D view context action or worker API) — position
   and orientation are independent; the camera never auto-aims.
   Terminal A itself can move: `set_terminal_motion(world, "drift",
   vx=0.3, yaw_rate_deg_s=0.2)` in the world-builder API gives the
   platform constant velocity + attitude rates (two satellites). The
   mount rides strapdown — the camera keeps world-frame pointing and
   the loop rejects platform motion as a pointing disturbance within
   the 5°/s gimbal authority. (This is real platform dynamics, not the
   Platform path combo under disturbances, which only fakes
   image-plane jitter.)
3. Add beacons at independent positions; assign each beacon its own
   motion model, parameters, and seed.
4. Choose one PRIMARY BEACON (starred in the beacon list). Changing it
   never moves other beacons or the camera.
5. Configure disturbances, then press START. START with no world is
   refused with a logged error.

### Controlling a beacon (simulation mode)

Only `user_controlled` beacons respond to operator input:

1. Trajectory combo → `user_controlled`, then **+ Add Beacon**.
2. Click the main window (not a spin box) and drive it with the
   keyboard: **A/D** ±X, **Q/E** ±Y, **W/S** ±Z, **arrow keys**,
   **Space** hold, **R** reset to spawn, **M** random maneuver.
   Steps are 10 world units per press.
3. Alternatively drag the beacon in the 3D world view (works for
   user-controlled beacons; scripted trajectories own their motion).
4. Pressing a movement key with no user-controlled beacon logs a
   warning telling you exactly what to do.

### Loading a video

Two equivalent ways, both land on the Camera Tracking page tracking
the file:

1. Top bar: press the cyan **LOAD VIDEO** button and pick an
   MP4/AVI/MOV/MKV/WebM file (bad files are rejected with a logged
   error before anything starts).
2. Top bar SOURCE combo → VIDEO (prompts immediately); cancelling
   reverts the combo instead of stranding the mode.

While a video runs, SEEK jumps to a frame; START with VIDEO mode and
no file prompts for one instead of failing.

Long files play through: a damaged packet mid-file no longer stops
the run — the decoder re-opens and seeks past the unreadable stretch
(logged as "recovered at frame N (skipped K)"), skipping further
ahead if the damage spans seconds. The run stops only at true
end-of-file ("Video ended after N frames (M total)") or, if the file
is truncated/damaged past recovery, with an ERROR naming the stall
frame and decoder position. Per-frame history is stored at 8
bytes/entry with capped event rings, so 24-hour files stay
resident-friendly.

### Reticle legend (camera tracking page)

The beacon gate is four square brackets with a + centroid marker:

| Color | Meaning |
|-------|---------|
| Green | Locked (Kalman TRACKING + quality gate) |
| Orange | Tracking/acquiring (estimate, not yet locked) |
| Yellow | Detected but unlocked |
| Red | Lost (no detection) |

The amber diamond with `PRED +0.10s` tag is the AI motion forecast;
dotted circles are estimate/prediction uncertainty. `TRK: <state>`
text (top-left) mirrors the same state machine as the telemetry grid
below the frame.

---

## 6. Benchmark Evaluation

### Running benchmarks

```bash
# Synthetic benchmark (closed-loop)
fsoc-sim -t circular -d 30 --output runs/bench_synthetic

# Video benchmark
fsoc-bench -i test_video.mp4 -v --output runs/bench_video

# Generated-world benchmark (replaces the retired numbered suite)
python -m fsoc_tracker.cli.run_benchmark sim --method kalman_expert \
    --world nominal --seed 42 --frames 300 --output runs/bench_world
```

### Output files

Each run produces:
- `metadata.json` — Session configuration and timing
- `errors.csv` — Per-frame tracking error
- `summary.txt` — Human-readable summary
- `sih_report.json` — (validation suite) Threshold evaluation

### Interpreting results

Key metrics:
- **Acquisition time** — Time to first lock (target: ≤ 2s)
- **RMSE** — Root-mean-square tracking error (target: ≤ 10px)
- **Loss percentage** — Fraction of frames with lost track (target: < 5%)
- **Reacquisition time** — Time to re-lock after loss (target: ≤ 1s)
- **Processing FPS** — Pipeline throughput (target: ≥ 20 FPS)
- **Precision / Recall** — Per-frame detection classification against
  ground truth (TP = detected within 10 px, FP = ghost/wrong target,
  FN = missed visible target), reported in benchmark JSON + tables.

---

## 7. Advanced Features

Enable in `configs/production.yaml`:

```yaml
advanced:
  enabled: true
  quality_analysis: true      # Image quality assessment
  uncertainty_estimation: true # Multi-source uncertainty
  adaptive_perception: true   # ROI-based detection
  fusion_enabled: true        # Classical+AI fusion
  adaptive_kalman: true       # Q/R adaptation
  maneuver_detection: true    # Motion classification
  search_controller: true     # Progressive search
  lock_quality: true          # Multi-factor lock score
  adaptive_control: true      # Gain scheduling
```

### Perception backends (PERC tab)

- **classical** (default, production): bright-spot detector, 100%
  detection at 0.32 px RMSE on the noisy reference video.
- **ai** (experimental, NOT recommended for tracking): trained
  BeaconCNN heatmap detector (`artifacts/models/beacon-numpy-v1`,
  weights verified at load). Measured on the same noisy video: 98.8%
  detection rate but 297 px RMSE — it fires confidently on noise
  structure, so the tracker follows ghosts. Useful for ablation study
  only.
- **hybrid** (experimental): classical+AI fusion, measured 230 px RMSE
  on the noisy video for the same reason. Missing weights fall back to
  classical with a logged warning (never silent random weights).

Backend switches apply live, mid-run, without rebuilding state. The
Kalman/AI-Brain toggles below are the recommended ablation tools;
they leave detection quality intact.

### Ablation toggles (PERC tab)

- **Kalman Tracker** off = raw-detection tracking (estimate follows
  detections, no filtering/prediction). Re-enabling resets the filter
  for a clean handoff.
- **AI Brain** off = mission policy, lead compensation, and failure
  prediction bypassed (HUD shows DISABLED); tracking/control continue.
- Use these to demonstrate each layer's contribution in one run.

---

## 8. Troubleshooting

### No detection in simulation

- Ensure target z > camera z (target must be in front of camera)
- Check perception threshold (`threshold_value` in config)
- Use `threshold_mode: global` with `threshold_value: 20.0` for simulation

### Low FPS

- Disable advanced features: `advanced.enabled: false`
- Reduce image resolution: `camera.width: 320, camera.height: 240`
- Use simpler trajectory: `-t straight_line`

### Video benchmark fails

- Ensure OpenCV is installed: `pip install opencv-python`
- Check video format is supported (MP4, AVI, MKV, MOV)
- Try: `fsoc-bench -i video.mp4 -v` for verbose output

---

## 9. Command Reference

| Command | Description |
|---------|-------------|
| `python -m fsoc_tracker` | Smoke test (real pipeline, 5s) |
| `python -m fsoc_tracker --gui` | Launch GUI |
| `python -m fsoc_tracker -c CONFIG` | Smoke test with config |
| `fsoc-sim` | Headless simulation |
| `fsoc-sim -t TRAJ -d SEC` | Simulation with trajectory and duration |
| `fsoc-bench -i VIDEO` | Video benchmark |
| `pytest` | Run all tests |
| `pytest --cov=fsoc_tracker` | Run tests with coverage |
| `ruff check src/ tests/` | Lint code |

---

## 10. Demonstration video

`artifacts/demo/` holds ready-to-play deliverables:

- `demo_app_5min.mp4` — 5-minute closed-loop application demo:
  simulated beacon on a sinusoidal 3D path (regular FOV exits and
  re-acquisitions) with the tracking reticle, prediction marker, and
  telemetry bar burned in. This is the optional 3–5 minute
  demonstration deliverable.
- `beacon_5min_3d.mp4` / `beacon_5min_inside.mp4` — clean 5-minute
  beacon footage (no overlays) for VIDEO-mode testing: feed either
  file to the tracker via LOAD VIDEO.
- `ps_compliance_4min.mp4` — 4-minute problem-statement compliance
  video: 16 back-to-back 15 s segments, one per parameter group
  (sizes, shapes, motions, multi-target, noises, atmosphere, FOV
  exit). Deliberately overlay-free — burned-in text would hand the
  tracker free static targets, so segment identity lives in the table
  below and in `scripts/generate_ps_compliance_video.py`.

| Seg | Time | Parameter under test |
|-----|------|----------------------|
| 1 | 0:00–0:15 | Baseline: 640×480, 4°×3°, 30 fps, 10 px spot |
| 2–3 | 0:15–0:45 | Target size 5 px / 20 px |
| 4–5 | 0:45–1:15 | Shape circular / gaussian spot |
| 6–10 | 1:15–2:30 | Motion circular / figure-8 / random / spiral / sinusoidal |
| 11 | 2:30–2:45 | Multi-target ×3 |
| 12–14 | 2:45–3:30 | Gaussian σ=20 / salt&pepper 10% / poisson + low light |
| 15 | 3:30–3:45 | Fog + rain + jitter ±20 px |
| 16 | 3:45–4:00 | FOV exit + re-acquisition (loss logging) |

Regenerate the app demo with:

```bash
.venv/bin/python scripts/generate_demo_video.py  # 30 s smoke demo
.venv/bin/python scripts/generate_app_demo.py    # 5 min deliverable
```
