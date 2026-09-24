# Benchmarking Guide

## Worlds (generated, seeded — no presets)

`--world`: `nominal` (single slow beacon), `multi` (3 candidates),
`distractor` (glint field), `coded` (primary blinks code 10110010,
five equal decoys blink the inverse — only the temporal code
separates them), `moving` (Terminal A itself drifts + yaws while the
beacon flies: two moving endpoints), `loss` (temporal outages), `noise`,
`fog`, `jitter`, `fast` (240 px/s lateral crosser for lag/lead
evaluation). Same world + seed reproduces exactly; `--seeds 42,43,44`
aggregates mean/std across seeds (Monte Carlo acquisition: the
`Acquisition (s)` mean/std row is the Tacq distribution).

## Tuning flags (opt-in experiments, defaults are the production path)

- `--assoc-method mahalanobis` — covariance-shaped association gate
  (default: nearest-neighbor). Measured: large RMSE win in glint
  fields, small clean-scene cost; use per scenario, not globally.
- `--assoc-gate-px N` — Euclidean gate override (default: 80).
- `--assoc-appearance W` — appearance-term weight in the association
  score (default: 0 off). Prefers candidates matching the track's
  running size/brightness signature when several pass the geometric
  gate. Measured on 30 seeded planted-glint trials: hijacks 24/30
  (W=0) → 0/30 (W=8); no effect on the 6-lookalike distractor world
  (geometry gaps dominate there).
- `--assoc-identity W` + `--identity-code CODE` — coded-identity term
  weight with the operator-configured binary beacon code (default: 0
  off / no code). Candidates whose blink history confidently mismatches
  the code are skipped before ranking, so the tracker coasts through
  the true beacon's OFF gaps instead of hijacking a decoy. Measured on
  `coded` (kalman_expert, seeds 42/7/13, 600f): RMSE 12.08→2.16 px,
  false-lock frames 54.8%→2.5% (72→2 events) at W=40, code 10110010.
  Retention reads lower (99.5%→51.5%) because a 50%-duty beacon yields
  photons half the time — the RMSE/false-lock pair is the honest
  comparison, retention is reported alongside, not hidden.
- `--lead` / `--lead-time S` — lead-angle compensation: aim ahead
  along estimated velocity (default off, lookahead 0.1 s, capped).
  Measured on `fast`: RMSE 5.56→4.68, loss 30.5%→22.5% at 0.2 s.
  The GUI worker enables lead automatically via the mission brain
  only for confident, safety-approved FAST_TARGET_MOTION.


## PS Evaluation Benchmark (closed-loop simulation)

Reproducible per-method commands. Every run executes the real
closed-loop pipeline and writes a JSON record + per-frame CSV log
automatically (default `logs/benchmark/`). The exact repro command is
stored in each JSON record (`repro_command`).

```bash
python -m fsoc_tracker.cli.run_benchmark sim --method kalman_expert --world multi --seed 42 --frames 300 --output logs/benchmark
python -m fsoc_tracker.cli.run_benchmark sim --method classical_pid --world multi --seed 42 --frames 300 --output logs/benchmark
python -m fsoc_tracker.cli.run_benchmark sim --method learned_temporal_expert --world multi --seed 42 --frames 300 --output logs/benchmark
python -m fsoc_tracker.cli.run_benchmark sim --method learned_temporal_learned_policy --world multi --seed 42 --frames 300 --output logs/benchmark
python -m fsoc_tracker.cli.run_benchmark sim --method full_ai_mission --world multi --seed 42 --frames 300 --output logs/benchmark
```

### Methods (only genuinely available ones are offered)

| Method | Stack |
|--------|-------|
| CLASSICAL + PID | Raw detection centroid drives a proportional law (no Kalman, no AI) |
| KALMAN + EXPERT | Kalman-filtered track drives PID (engineered stack) |
| LEARNED TEMPORAL + EXPERT | Kalman + PID + trained linear motion model via the mission brain (expert policy) |
| LEARNED TEMPORAL + LEARNED POLICY | Learned motion + trained policy classifier under the safety envelope |
| FULL AI MISSION | Production TrackingPipeline with the full adaptive stack |

Availability is probed at runtime from the trained artifacts
(`artifacts/models/mission-v3/`). A method whose weights are missing
or stale is reported NOT AVAILABLE and refused — never executed with
a stand-in. Notes:

- The learned situation classifier is NOT used anywhere: its saved
  weights predate the current 10-category situation set, so the
  expert classifier always handles situation assessment.
- The GRU temporal checkpoint is experimental and is NOT used by any
  benchmark method; the linear motion model (trained/evaluated, test
  RMSE recorded in `experiment.json`) is the learned temporal
  component.

### Per-run record

Each JSON record contains: method, scenario, seed, source, duration,
FPS, processing latency (mean/p95), acquisition time, RMSE, MAE, P95,
maximum error, target loss %, reacquisition time, lock retention,
FOV retention, search duration, link downtime, message delivery
(delivered/total), safety events, AI action histogram, learned-motion
prediction RMSE (AI modes), provenance (harness, model versions,
experiment metrics), and the repro command. The sibling
`*_frames.csv` holds every frame (detection, track, error, camera,
link, AI decision, latency).

### Ground-truth discipline

Ground truth is used for SCORING ONLY (error metrics, FOV retention,
link geometry). Perception, tracking, control, and the mission brain
see only the rendered image. VIDEO/LIVE sources are therefore NOT
offered for PS benchmarking (monocular input has no depth/ground
truth); the GUI disables them.

## Quick Start

### Synthetic Benchmark

```python
from fsoc_tracker.benchmark.run import ci_regression_benchmark

result = ci_regression_benchmark()
print(result.summary_table())
```

### External Video Benchmark

```python
from fsoc_tracker.benchmark.engine import BenchmarkEngine
from fsoc_tracker.benchmark.video_source import VideoBenchmarkSource

with VideoBenchmarkSource("input.mp4") as source:
    engine = BenchmarkEngine()
    result = engine.run_source(source)
    print(result.summary_table())
```

### Full Pipeline with Export

```python
from fsoc_tracker.benchmark.engine import BenchmarkEngine
from fsoc_tracker.benchmark.video_source import VideoBenchmarkSource
from fsoc_tracker.benchmark.export import export_json, export_csv
from fsoc_tracker.benchmark.report import generate_html_report
from fsoc_tracker.benchmark.plots import generate_plots

with VideoBenchmarkSource("input.mp4") as source:
    engine = BenchmarkEngine()
    result = engine.run_source(source)

export_json(result, "output/benchmark_result.json")
export_csv(engine.get_collector().frames, "output/frame_metrics.csv")
generate_html_report(result, "output/report.html")
generate_plots(result, engine.get_collector().frames, "output/plots")
```

## Supported Video Formats

| Format | Support |
|--------|---------|
| .mp4 | Full (H.264, H.265) |
| .avi | Full |
| .mkv | Full |
| .mov | Full |
| Other | Depends on OpenCV build |

## FPS Independence

The system works at any frame rate. FPS is never hardcoded.

```python
# Works at any FPS
source_5fps = VideoBenchmarkSource("slow.mp4")    # 5 FPS
source_60fps = VideoBenchmarkSource("fast.mp4")    # 60 FPS
source_var = VideoBenchmarkSource("variable.mp4")  # Variable FPS
```

Timestamps are preferred over frame_index / FPS.

## Ground Truth

### Without Ground Truth (default)

Accuracy metrics report NOT_EVALUABLE. Detection rate, tracking stability, latency, and FPS are still measured.

### With Sidecar Ground Truth

Create a JSON file alongside the video:

```json
[
  {"frame_index": 0, "x": 320, "y": 240, "timestamp": 0.0},
  {"frame_index": 1, "x": 325, "y": 238, "timestamp": 0.033},
  ...
]
```

Or as JSONL:
```
{"frame_index": 0, "x": 320, "y": 240}
{"frame_index": 1, "x": 325, "y": 238}
```

```python
from fsoc_tracker.benchmark.ground_truth import SidecarGroundTruthProvider

gt = SidecarGroundTruthProvider("video_ground_truth.json")
engine = BenchmarkEngine(ground_truth=gt)
```

### With Synthetic Ground Truth

```python
from fsoc_tracker.benchmark.ground_truth import SyntheticGroundTruthProvider, GroundTruthPoint

gt = SyntheticGroundTruthProvider()
gt.add_point(GroundTruthPoint(frame_index=0, target_x=320, target_y=240))
gt.add_point(GroundTruthPoint(frame_index=1, target_x=325, target_y=238))
```

## SIH Thresholds

Default SIH26169 thresholds:
- Acquisition time: ≤ 2.0s
- Tracking error: ≤ 10.0px
- Target loss: < 5.0%
- Re-acquisition: ≤ 1.0s
- Processing FPS: ≥ 20.0

Override with custom thresholds:

```python
from fsoc_tracker.benchmark.models import BenchmarkThresholds

t = BenchmarkThresholds(acquisition_max_s=3.0, tracking_error_max_px=15.0)
```

## Result Inspection

### Summary Table

```python
print(result.summary_table())
```

### JSON Export

```python
import json
with open("benchmark_result.json") as f:
    data = json.load(f)
print(data["metrics"]["tracking"]["rmse_px"])
```

### Frame-Level Data

```python
for fm in engine.get_collector().frames:
    if fm.error_px is not None:
        print(f"Frame {fm.frame_index}: error={fm.error_px:.2f}px")
```

### Recompute-verify (reports prove themselves)

Every GUI-run performance report stores the full per-frame raw error
series next to the JSON (`*_raw_errors.csv`, path recorded in the
report). Recompute every aggregate from raw and compare:

```bash
python -m fsoc_tracker.cli.run_benchmark verify artifacts/reports/report_<ts>.json
# VERIFY: PASS — or FAIL with per-metric stored-vs-recomputed diffs
```

A tampered headline number fails verification; a missing raw file
refuses it. The GUI logs the verdict automatically at stop time.

## CI Regression

Run a quick deterministic benchmark for regression testing:

```python
from fsoc_tracker.benchmark.run import ci_regression_benchmark

result = ci_regression_benchmark()
assert result.frame_count == 150
assert result.performance.throughput_fps > 0
```

## Benchmark Demo

```bash
python -m fsoc_tracker.benchmark_demo --output benchmark_output --frames 300 --fps 30
```

Generates: `benchmark_result.json`, `frame_metrics.csv`, `event_log.csv`, `report.html`, `plots/`
