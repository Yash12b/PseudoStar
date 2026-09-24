# FSOC Tracker

AI-Based Virtual Camera Tracking System for Coarse Alignment of Mobile Free Space Optical Communication (FSOC) Terminals.

**SIH Problem Statement: SIH26169**

## Overview

This system implements a complete simulation and tracking pipeline for FSOC terminal coarse alignment, including:

- Configurable virtual environment (2000x2000+ canvas)
- Moving beacon targets with multiple motion patterns
- Virtual pan/tilt camera control
- Target detection (classical CV + AI)
- Continuous tracking with state estimation
- Configurable disturbances and noise models
- Real-time performance metrics
- External video and live camera support
- Benchmark evaluation engine

### Intelligence Layer

The system includes a complete AI intelligence layer that enhances tracking while never bypassing safety:

- **Temporal Prediction**: GRU-based motion prediction with 5-horizon displacement
- **Situation Awareness**: 7-state classifier driving strategy selection
- **Failure Prediction**: Logistic regression early warning for track loss
- **Adaptive ROI**: Dynamic region-of-interest based on uncertainty and risk
- **Mission Brain**: Full AI decision pipeline (observe → classify → decide → approve → execute)
- **Explainability**: Human-readable decision explanations with feature contributions
- **Adaptive Kalman**: Automatic Q/R scaling from tracking metrics
- **Adaptive Controller**: Dynamic PID gains from error/velocity telemetry
- **Search Controller**: Intelligent reacquisition when target is lost

Safety always dominates AI — deterministic fallback, geometry, and rate limits are never bypassed.

## Quick Start

```bash
# Install the desktop application and developer tools
pip install -e ".[dev,desktop]"

# Launch the standalone desktop GUI
python -m fsoc_tracker

# Run the headless smoke test explicitly
python -m fsoc_tracker --smoke

# Run headless simulation
fsoc-sim -t circular -d 10 --disturbance light

# Run video benchmark
fsoc-bench -i video.mp4 -v

# Run a generated-world benchmark (deterministic, seeded)
python -m fsoc_tracker.cli.run_benchmark sim --method kalman_expert --world multi --seed 42 --frames 300 --output logs/benchmark

# Multi-seed aggregation (mean/std across seeds)
python -m fsoc_tracker.cli.run_benchmark sim --method kalman_expert --world loss --seeds 42,43,44 --frames 300 --output logs/benchmark

# Run with custom config
python -m fsoc_tracker -c configs/benchmark.yaml

# Build an offline standalone executable (macOS/Linux; build Windows on Windows)
python scripts/build_desktop.py
# Output: dist/FSOC-Tracker (or FSOC-Tracker.exe on Windows)

# Run tests
pytest
```

## Architecture

See [docs/architecture/system-overview.md](docs/architecture/system-overview.md) for the complete system design.

### Pipeline

```
FrameSource → Frame → Perception → Tracking → Control → VirtualPTZ → next frame
                         ↓            ↓          ↓
                    Adaptive ROI  Adaptive Kalman  Adaptive Controller
                         ↓            ↓          ↓
                    Mission Brain → Situation + Failure Prediction → Temporal Prediction
                         ↓
                    Explainability + Decision Logging
```

### 18-Stage Architecture

1. **Stage 1-14**: Core simulation, perception, tracking, control, GUI, CLI, benchmark, testing
2. **Stage 15**: Full audit and gap analysis
3. **Stage 16**: Temporal observable dataset generator
4. **Stage 17**: Retrain mission models on real observable data
5. **Stage 18**: Train temporal GRU predictor
6. **Stage 19**: Build failure predictor
7. **Stage 20**: Build adaptive ROI module
8. **Stage 21**: Wire mission brain into TrackingPipeline
9. **Stage 22**: Integrate adaptive Kalman filter
10. **Stage 23**: Integrate adaptive controller
11. **Stage 24**: Integrate search controller
12. **Stage 25**: Explainability layer
13. **Stage 26**: Closed-loop AI benchmark
14. **Stage 27**: Integration tests
15. **Stage 28**: SIH validation
16. **Stage 29**: Documentation update
17. **Stage 30**: Final test pass
18. **Stage 31**: Package & release

### Key Design Principles

1. **Source-agnostic**: Works with synthetic simulation, video files, live cameras, and streams
2. **Never hardcode FPS**: All timing derived from frame timestamps
3. **Typed everything**: Dataclasses and enums throughout, no magic numbers
4. **Configurable**: YAML-based configuration with Pydantic validation
5. **Modular**: Clean interfaces allow independent subsystem development

## Project Structure

```
fsoc_tracker/
├── configs/              # YAML configuration profiles
│   ├── default.yaml      # Standard SIH26169 defaults
│   ├── production.yaml   # All advanced features enabled
│   ├── baseline.yaml     # Minimal (no advanced features)
│   ├── debug.yaml        # Full diagnostics
│   └── benchmark.yaml    # Video benchmark config
├── src/fsoc_tracker/
│   ├── core/             # Domain models, interfaces, time, exceptions
│   ├── config/           # Pydantic configuration system
│   ├── io/               # Frame source adapters
│   ├── perception/       # Target detection (classical + AI + hybrid)
│   ├── tracking/         # Kalman tracking, state machine, search
│   ├── control/          # PID + adaptive coarse pointing control
│   ├── simulation/       # Virtual 3D environment, camera, sensor
│   ├── disturbances/     # Noise, atmosphere, turbulence models
│   ├── ai/               # CNN perception, training, export, intelligence layer
│   ├── benchmark/        # Evaluation engine, metrics, reports
│   ├── gui/              # PySide6 aerospace HUD
│   ├── pipeline/         # Authoritative pipeline orchestrator
│   ├── cli/              # CLI runners (simulation, benchmark)
│   ├── advanced/         # Adaptive intelligence, diagnostics
│   └── app/              # Entry points
├── tests/                # 1420+ unit + integration tests
├── docs/                 # Architecture, algorithms, development
├── artifacts/            # Trained models and datasets
│   ├── models/           # mission/policy/failure/temporal weights (tracked)
│   └── datasets/         # temporal splits (tracked); large beacon image
│                         # datasets are synthetic and regenerable (see below)
└── scripts/              # Utility scripts (incl. demo video generator)
```

## Configuration

All parameters are configurable via YAML files. Reference values from SIH26169 PS:

| Parameter | Default | SIH Reference |
|-----------|---------|---------------|
| Camera resolution | 640x480 | 640x480 |
| Camera FOV | 4° x 3° | 4° x 3° |
| Camera update rate | 30 Hz | 30 Hz min |
| Target size | 10 px | ~10x10 px |
| Max pan/tilt speed | 5°/s | 5°/s |
| Control update rate | 20 Hz | 20 Hz min |
| Acquisition limit | 2.0 s | 2.0 s |
| Tracking error limit | 10 px | 10 px |
| Processing FPS | 20 FPS | 20 FPS min |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest --cov=fsoc_tracker

# Lint
ruff check src/ tests/

# Type check
ruff check --select=ANN src/
```

## Regenerating large artifacts

The multi-GB synthetic image datasets and the experimental
heavy-training checkpoint are intentionally **not** stored in git.
Regenerate them locally:

```bash
# Beacon image datasets (seeds fixed for reproducibility)
python -m fsoc_tracker.ai.generate_dataset --output artifacts/datasets/beacon-v3 \
    --samples 10000 --validation-samples 1000 --seed 2026
python -m fsoc_tracker.ai.train_visual --output artifacts/models/beacon-visual-v2/checkpoint.pt \
    --samples 3000 --validation-samples 400 --seed 2026 --epochs 5 --batch-size 32
```

All tracked model weights (`mission-*`, `failure-v1`, `temporal-*`,
`beacon-visual-v1`, `disturbance-v1`) and the small temporal datasets
needed by the test suite ship with the repo.

## Roadmap

See [docs/development/roadmap.md](docs/development/roadmap.md) for the full development roadmap.

## License

MIT
