"""Headless benchmark runner.

Reproducible PS-metric commands:

    python -m fsoc_tracker.cli.run_benchmark sim --method kalman_expert \\
        --world multi --seed 42 --frames 300 --output logs/benchmark
    python -m fsoc_tracker.cli.run_benchmark video --input video.mp4 --output runs/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from fsoc_tracker.control.controller import CameraActuator, CoarsePointingController
from fsoc_tracker.perception.config import PerceptionConfig
from fsoc_tracker.pipeline.pipeline import TrackingPipeline
from fsoc_tracker.pipeline.session import SessionController
from fsoc_tracker.pipeline.sources import VideoSource
from fsoc_tracker.tracking.tracker import KalmanTracker


def run_video_benchmark(
    video_path: str,
    output_dir: str = "runs",
    verbose: bool = False,
    backend: str = "classical",
) -> dict:
    """Run benchmark on an external video file."""
    from fsoc_tracker.perception.backends import build_backend
    print(f"[BENCH] input={video_path} backend={backend}")

    source = VideoSource(video_path)
    detector, effective, _ = build_backend(
        backend, PerceptionConfig.for_video())
    print(f"[BENCH] perception backend in effect: {effective}")
    pipeline = TrackingPipeline(
        # Video input gets video-tuned perception (same config family
        # the GUI worker applies in VIDEO mode), not sim defaults.
        perception=detector,
        tracker=KalmanTracker(),
        controller=CoarsePointingController(),
        actuator=CameraActuator(),
    )
    pipeline.set_source(source)

    session = SessionController(pipeline, output_dir=output_dir)
    session.configure(
        source_type="video",
        source_info=video_path,
    )

    session.start()
    frame_count = 0
    t_wall_start = time.monotonic()

    while session.state.value == "RUNNING":
        result = session.step()
        if result is None:
            break
        frame_count += 1
        if verbose and frame_count % 30 == 0:
            print(
                f"  frame={frame_count:5d} t={result.timestamp_s:7.2f}s "
                f"det={result.perception.detected if result.perception else False} "
                f"track={result.tracking.state.name if result.tracking else '?'}"
            )

    final = session.stop()
    wall_time = time.monotonic() - t_wall_start

    print("\n" + "=" * 60)
    print(final.summary())
    print(f"Wall-clock time: {wall_time:.2f}s")
    print(f"Artifacts: {final.artifacts_dir}")
    print("=" * 60)

    return {
        "session_id": final.metadata.session_id,
        "frame_count": final.frame_count,
        "duration_s": final.duration_s,
        "processing_fps": final.processing_fps,
        "artifacts_dir": final.artifacts_dir,
    }


def run_sim_benchmark(
    method: str,
    world: str,
    seed: int,
    frames: int,
    output: str,
    verbose: bool = False,
    seeds: list[int] | None = None,
    assoc_gate_px: float | None = None,
    assoc_method: str | None = None,
    assoc_appearance: float | None = None,
    assoc_identity: float | None = None,
    identity_code: str | None = None,
    lead_compensation: bool = False,
    lead_time_s: float | None = None,
) -> dict:
    """Run a closed-loop simulation benchmark for one method.

    ``world`` names a generated benchmark world profile (see
    ``simulation.world_builder``); ``seed`` makes the run reproducible.
    With several seeds each runs fully and independently, then per-metric
    mean/std aggregation is printed and saved.
    """
    from fsoc_tracker.benchmark.methods import METHOD_LABELS
    from fsoc_tracker.benchmark.runner import (
        BenchmarkMode,
        BenchmarkRunConfig,
        BenchmarkRunner,
    )

    seed_list = list(seeds) if seeds else [seed]
    try:
        mode = BenchmarkMode(method)
    except ValueError:
        valid = ", ".join(m.value for m in BenchmarkMode)
        print(f"ERROR: unknown method '{method}'. Valid: {valid}", file=sys.stderr)
        sys.exit(2)

    runner = BenchmarkRunner()

    def log_callback(level: str, msg: str) -> None:
        if verbose or level in ("WARN", "ERROR"):
            print(f"[{level}] {msg}")

    per_seed: list[dict] = []
    last_metrics = None
    for s in seed_list:
        config = BenchmarkRunConfig(
            mode=mode, world_profile=world, seed=s, max_frames=frames,
            output_dir=output, assoc_gate_px=assoc_gate_px,
            assoc_method=assoc_method, assoc_appearance=assoc_appearance,
            assoc_identity=assoc_identity, identity_code=identity_code,
            lead_compensation=lead_compensation,
            lead_time_s=lead_time_s,
        )
        print(f"[BENCH] method={method} world={world} seed={s} frames={frames}")
        try:
            last_metrics = runner.run(config, log_callback=log_callback)
        except Exception as e:  # noqa: BLE001 — includes MethodUnavailableError
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(3)
        per_seed.append(last_metrics.to_dict())
        print()
        print(f"=== {METHOD_LABELS.get(method, method)} (seed {s}) ===")
        print(last_metrics.summary_table())

    if len(per_seed) == 1:
        d = per_seed[0]
        print(f"JSON log: {last_metrics.log_json_path}")
        print(f"CSV log:  {last_metrics.log_csv_path}")
        print(f"Repro:    {last_metrics.repro_command}")
        return d

    return _aggregate_seed_runs(method, world, frames, output, per_seed)


def _aggregate_seed_runs(
    method: str,
    world: str,
    frames: int,
    output: str,
    per_seed: list[dict],
) -> dict:
    """Aggregate multi-seed runs: mean/std per metric + JSON artifact."""
    import json
    import statistics
    from pathlib import Path

    def column(key: str) -> list[float]:
        return [float(r[key]) for r in per_seed
                if r.get(key) is not None]

    def stat(key: str) -> dict:
        vals = column(key)
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }

    agg = {
        "method": method,
        "world": world,
        "frames": frames,
        "seeds": [r["seed"] for r in per_seed],
        "acquisition_s": stat("acquisition_s"),
        "rmse_px": stat("rmse_px"),
        "mae_px": stat("mae_px"),
        "p95_px": stat("p95_px"),
        "max_error_px": stat("max_error_px"),
        "loss_rate_pct": stat("loss_rate_pct"),
        "reacquisition_s": stat("reacquisition_s"),
        "lock_retention_pct": stat("lock_retention_pct"),
        "fps": stat("fps"),
        "frames_processed": sum(int(r["frames_processed"]) for r in per_seed),
    }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    seeds_tag = "-".join(str(s) for s in agg["seeds"])
    agg_path = out / f"aggregate_{method}_world{world}_seeds{seeds_tag}.json"
    agg_path.write_text(json.dumps(agg, indent=2))

    print()
    print(f"=== {method} × {len(per_seed)} seeds (world {world}) ===")
    for key, label in [
        ("acquisition_s", "Acquisition (s)"),
        ("rmse_px", "RMSE (px)"),
        ("loss_rate_pct", "Loss (%)"),
        ("reacquisition_s", "Reacquisition (s)"),
        ("lock_retention_pct", "Lock retention (%)"),
        ("fps", "FPS"),
    ]:
        st = agg[key]
        if st["mean"] is None:
            print(f"{label:<20s}: N/A (no events)")
        else:
            print(f"{label:<20s}: {st['mean']:.3f} ± {st['std']:.3f} (n={st['n']})")
    print(f"Aggregate JSON: {agg_path}")
    agg["aggregate_json_path"] = str(agg_path)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="FSOC Benchmark Runner")
    # Legacy flat form: fsoc-bench -i video.mp4 (equivalent to `video`).
    parser.add_argument(
        "-i", "--input", default=None, help="Path to video file (legacy video form)"
    )
    parser.add_argument("-o", "--output", default="runs")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=False)

    p_sim = sub.add_parser("sim", help="Closed-loop simulation benchmark (PS metrics)")
    p_sim.add_argument("--method", default="kalman_expert",
                       help="classical_pid | kalman_expert | learned_temporal_expert | "
                            "learned_temporal_learned_policy | full_ai_mission")
    p_sim.add_argument("--world", default="multi",
                       help="Generated benchmark world profile: nominal | "
                            "multi | distractor | coded | moving | loss | "
                            "noise | fog | jitter | fast")
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.add_argument("--seeds", default=None,
                       help="Comma-separated seeds, e.g. '42,43,44' "
                            "(overrides --seed; aggregates mean/std)")
    p_sim.add_argument("--frames", type=int, default=300)
    p_sim.add_argument("--assoc-gate-px", type=float, default=None,
                       help="Tracker association gate in px (default: 80)")
    p_sim.add_argument("--assoc-method", default=None,
                       help="nearest_neighbor | mahalanobis (default: nearest)")
    p_sim.add_argument("--assoc-appearance", type=float, default=None,
                       help="Appearance term weight vs distractor hijack (default: 0 off)")
    p_sim.add_argument("--assoc-identity", type=float, default=None,
                       help="Coded-identity term weight vs code-mismatched "
                            "decoys (default: 0 off; needs --identity-code)")
    p_sim.add_argument("--identity-code", default=None,
                       help="Operator-configured binary beacon code, e.g. "
                            "'10110010' (coded world primary uses 10110010, "
                            "decoys use the inverse)")
    p_sim.add_argument("--lead", action="store_true",
                       help="Enable lead-angle compensation (aim ahead "
                            "along velocity; experimental, off by default)")
    p_sim.add_argument("--lead-time", type=float, default=None,
                       help="Lead lookahead in seconds (default: 0.1)")
    p_sim.add_argument("--output", default="logs/benchmark")
    p_sim.add_argument("-v", "--verbose", action="store_true")

    p_vid = sub.add_parser("video", help="External video benchmark")
    p_vid.add_argument("-i", "--input", required=True, help="Path to video file")
    p_vid.add_argument("-o", "--output", default="runs")
    p_vid.add_argument("-v", "--verbose", action="store_true")
    p_vid.add_argument("--backend", default="classical",
                       choices=["classical", "ai", "hybrid"])
    p_verify = sub.add_parser(
        "verify", help="Recompute a performance report from its raw CSV")
    p_verify.add_argument("report", help="Path to performance_report.json")
    args = parser.parse_args()

    if args.command == "verify":
        from fsoc_tracker.benchmark.report import verify_performance_report
        verdict = verify_performance_report(args.report)
        print(f"Report: {verdict.get('report')}")
        print(f"Raw frames checked: {verdict.get('frames', 0)}")
        for key, d in verdict.get("details", {}).items():
            mark = "OK " if d["match"] else "DIFF"
            print(f"  [{mark}] {key}: stored={d['stored']} recomputed={d['recomputed']}")
        print("VERIFY: " + ("PASS" if verdict["ok"] else "FAIL"))
        sys.exit(0 if verdict["ok"] else 1)

    if args.command == "sim":
        seed_list = None
        if getattr(args, "seeds", None):
            seed_list = [int(s) for s in str(args.seeds).split(",") if s.strip()]
        result = run_sim_benchmark(
            args.method, args.world, args.seed, args.frames, args.output,
            args.verbose, seeds=seed_list,
            assoc_gate_px=getattr(args, "assoc_gate_px", None),
            assoc_method=getattr(args, "assoc_method", None),
            assoc_appearance=getattr(args, "assoc_appearance", None),
            assoc_identity=getattr(args, "assoc_identity", None),
            identity_code=getattr(args, "identity_code", None),
            lead_compensation=bool(getattr(args, "lead", False)),
            lead_time_s=getattr(args, "lead_time", None),
        )
        sys.exit(0 if result["frames_processed"] > 0 else 1)

    # `video` subcommand or legacy flat `-i` form.
    video_input = getattr(args, "input", None)
    if args.command is None and video_input is None:
        parser.print_help()
        sys.exit(2)
    if not Path(video_input).exists():
        print(f"ERROR: Video file not found: {video_input}", file=sys.stderr)
        sys.exit(1)

    result = run_video_benchmark(video_input, args.output, args.verbose,
                                 backend=getattr(args, "backend", "classical"))
    sys.exit(0 if result["frame_count"] > 0 else 1)


if __name__ == "__main__":
    main()
