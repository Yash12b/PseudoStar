"""Performance report generator — mandatory deliverable.

Generates comprehensive performance logs containing:
- Simulation duration, FPS, acquisition time
- Average and maximum tracking error
- Lock retention rate
- Processing time
- Disturbance conditions
- Scene configuration
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def generate_performance_report(
    state: Any,
    output_dir: str = ".",
    filename: str = "performance_report.json",
    raw_errors: list[float] | None = None,
) -> str:
    """Generate a comprehensive performance report from the current state.

    ``raw_errors`` (optional): the FULL per-frame error list. When
    supplied it is stored alongside the report (sibling CSV) and its
    path recorded in the JSON, so verify_performance_report() can
    recompute every aggregate from raw data later. The in-JSON
    error_history keeps only the last 100 points for display.

    Returns the path to the generated report.
    """
    errors = state.errors
    perf = state.performance
    track = state.tracking
    sc = state.scorecard
    cam = state.camera
    ai = state.ai_state
    dist = state.disturbances
    link = state.optical_link

    # Compute aggregate metrics
    error_list = list(errors.errors_euclidean) if errors.errors_euclidean else []
    avg_error = sum(error_list) / len(error_list) if error_list else 0.0
    max_error = max(error_list) if error_list else 0.0
    min_error = min(error_list) if error_list else 0.0

    # P95 error
    sorted_errors = sorted(error_list)
    p95_idx = int(len(sorted_errors) * 0.95) if sorted_errors else 0
    p95_error = sorted_errors[min(p95_idx, len(sorted_errors) - 1)] if sorted_errors else 0.0

    # RMSE
    import math
    mse = sum(e ** 2 for e in error_list) / len(error_list) if error_list else 0.0
    rmse = math.sqrt(mse)

    # Lock retention (preferred: worker-maintained scorecard counter;
    # fallback preserves the old estimate path)
    total_frames = perf.frames_processed
    scorecard_retention = getattr(getattr(state, "scorecard", None),
                                  "lock_retention_pct", None)
    if scorecard_retention is not None:
        lock_retention = float(scorecard_retention)
        tracked_frames = None
    else:
        tracked_frames = state._worker._total_frames_tracked if hasattr(state, '_worker') and hasattr(state._worker, '_total_frames_tracked') else 0
        lock_retention = (tracked_frames / max(total_frames, 1)) * 100.0

    # FPS stats
    fps_list = list(errors.fps_history) if errors.fps_history else []
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0.0
    min_fps = min(fps_list) if fps_list else 0.0

    report = {
        "report_type": "FSOC Tracker Performance Report",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "simulation": {
            "duration_s": round(state.elapsed_s, 2),
            "total_frames": total_frames,
            "avg_fps": round(avg_fps, 1),
            "min_fps": round(min_fps, 1),
            "avg_processing_ms": round(perf.processing_ms, 2),
            "avg_perception_ms": round(perf.perception_ms, 2),
        },
        "tracking": {
            "final_state": track.state,
            "locked": track.locked,
            "avg_error_px": round(avg_error, 2),
            "max_error_px": round(max_error, 2),
            "min_error_px": round(min_error, 2),
            "rmse_px": round(rmse, 2),
            "p95_error_px": round(p95_error, 2),
            "lock_retention_pct": round(lock_retention, 1),
        },
        "sih_scorecard": {
            "acquisition_s": sc.acquisition_s,
            "acquisition_threshold_s": sc.acquisition_threshold,
            "acquisition_pass": (sc.acquisition_s is not None and sc.acquisition_s <= sc.acquisition_threshold),
            "rmse_px": round(rmse, 2),
            "rmse_threshold_px": sc.rmse_threshold,
            "rmse_pass": rmse <= sc.rmse_threshold,
            "loss_percent": round(sc.loss_percent, 2) if sc.loss_percent is not None else None,
            "loss_threshold_pct": sc.loss_threshold,
            "loss_pass": (sc.loss_percent is not None and sc.loss_percent <= sc.loss_threshold),
            "reacq_s": sc.reacq_s,
            "reacq_threshold_s": sc.reacq_threshold,
            "reacq_pass": (sc.reacq_s is not None and sc.reacq_s <= sc.reacq_threshold),
            "fps": round(avg_fps, 1),
            "fps_threshold": sc.fps_threshold,
            "fps_pass": avg_fps >= sc.fps_threshold,
        },
        "camera": {
            "hfov_deg": cam.hfov_deg,
            "vfov_deg": cam.vfov_deg,
            "resolution": f"{cam.image_width}x{cam.image_height}",
            "pan_deg": round(cam.pan_deg, 2),
            "tilt_deg": round(cam.tilt_deg, 2),
        },
        "disturbances": {
            "enabled": dist.enabled,
            "profile": dist.profile,
            "noise_type": dist.noise_type,
            "noise_sigma": dist.noise_sigma,
            "fog": dist.fog,
            "haze": dist.haze,
            "rain": dist.rain,
            "jitter_px": dist.jitter_px,
        },
        "optical_link": {
            "status": link.status,
            "range_m": round(link.range_m, 1),
            "beam_alignment_pct": round(link.beam_alignment_percent, 1),
            "angular_error_deg": round(link.angular_error_deg, 3),
            "link_quality_pct": round(link.link_quality_percent, 1),
        },
        "ai": {
            "situation": ai.situation,
            "action": ai.action,
            "confidence": round(ai.confidence, 3),
            "explanation": ai.explanation,
            "fallback_active": ai.fallback_active,
            "search_active": ai.search_active,
        },
        "handoff": {
            "ready": bool(getattr(state, "handoff_ready", False)),
            "stable_s": round(float(getattr(state, "handoff_stable_s", 0.0)), 3),
        },
        "error_history": {
            "timestamps": [round(t, 3) for t in errors.timestamps[-100:]],
            "errors_euclidean": [round(e, 2) for e in errors.errors_euclidean[-100:]],
            "fps_history": [round(f, 1) for f in errors.fps_history[-100:]],
        },
        "raw_errors_path": None,
    }

    out_path = Path(output_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_errors is not None:
        raw_path = out_path.with_name(out_path.stem + "_raw_errors.csv")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write("frame,error_px\n")
            for i, e in enumerate(raw_errors):
                f.write(f"{i},{float(e):.6f}\n")
        report["raw_errors_path"] = str(raw_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return str(out_path)


def recompute_error_aggregates(raw_errors: list[float]) -> dict[str, float]:
    """Recompute tracking aggregates from raw per-frame errors."""
    import math
    vals = [float(e) for e in raw_errors]
    if not vals:
        return {"avg_error_px": 0.0, "max_error_px": 0.0,
                "min_error_px": 0.0, "rmse_px": 0.0, "p95_error_px": 0.0}
    ordered = sorted(vals)
    return {
        "avg_error_px": sum(vals) / len(vals),
        "max_error_px": max(vals),
        "min_error_px": min(vals),
        "rmse_px": math.sqrt(sum(e ** 2 for e in vals) / len(vals)),
        "p95_error_px": ordered[min(int(len(ordered) * 0.95),
                                    len(ordered) - 1)],
    }


def verify_performance_report(report_path: str,
                              tolerance: float = 0.011) -> dict[str, Any]:
    """Recompute a report's aggregates from its raw CSV and compare.

    Returns {"ok": bool, "details": {...}, "report": path}. ``ok`` is
    True only when every recomputed aggregate matches the stored value
    within tolerance (covers the JSON round(x, 2) quantization).
    Missing raw file -> {"ok": False, "reason": ...}.
    """
    path = Path(report_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    raw_ref = (report.get("raw_errors_path")
               or str(path.with_name(path.stem + "_raw_errors.csv")))
    raw_path = Path(raw_ref)
    if not raw_path.is_absolute():
        raw_path = path.parent / raw_path.name
    if not raw_path.exists():
        return {"ok": False, "report": str(path),
                "reason": f"raw errors file missing: {raw_path}"}
    vals: list[float] = []
    with open(raw_path, encoding="utf-8") as f:
        header = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if header:
                header = False
                continue
            try:
                vals.append(float(line.split(",")[1]))
            except (IndexError, ValueError):
                continue
    recomputed = recompute_error_aggregates(vals)
    stored = report.get("tracking", {})
    details: dict[str, Any] = {}
    ok = True
    for key in ("avg_error_px", "max_error_px", "min_error_px",
                "rmse_px", "p95_error_px"):
        s = stored.get(key)
        r = recomputed[key]
        match = s is not None and abs(float(s) - r) <= tolerance
        details[key] = {"stored": s, "recomputed": round(r, 4),
                        "match": bool(match)}
        ok = ok and bool(match)
    return {"ok": ok, "report": str(path), "raw": str(raw_path),
            "frames": len(vals), "details": details}


def generate_html_report(result: Any, output_path: str) -> str:
    """Generate an HTML performance report from a BenchmarkResult.
    
    Returns the path to the generated HTML file.
    """
    acq = result.acquisition
    trk = result.tracking
    los = result.loss
    rea = result.reacquisition
    perf = result.performance

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>SIH26169 — FSOC Tracker Performance Report</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0a0e17; color: #e8f0f5; margin: 0; padding: 20px; }}
h1 {{ color: #22d3ee; border-bottom: 2px solid #22d3ee; padding-bottom: 10px; }}
h2 {{ color: #ff6b35; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #20303d; padding: 8px 12px; text-align: left; }}
th {{ background: #141e28; color: #22d3ee; }}
td {{ background: #0d141c; }}
.pass {{ color: #00ff88; font-weight: bold; }}
.fail {{ color: #ff4444; font-weight: bold; }}
.metric {{ font-size: 18px; font-weight: bold; color: #e8f0f5; }}
</style></head><body>
<h1>SIH26169 — FSOC Virtual Camera Tracking System</h1>
<h2>Performance Report</h2>
<p>Generated: {result.session.timestamp_s if hasattr(result.session, 'timestamp_s') else 'N/A'}</p>

<h2>Acquisition</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Acquisition Time</td><td class="metric">{f'{acq.stable_acquisition_s:.3f}s' if acq.stable_acquisition_s is not None else 'N/A'}</td></tr>
</table>

<h2>Tracking Performance</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>RMSE</td><td class="metric">{f'{trk.rmse_px:.2f}px' if trk.rmse_px is not None else 'N/A'}</td></tr>
<tr><td>P95 Error</td><td class="metric">{f'{trk.p95_error_px:.2f}px' if trk.p95_error_px is not None else 'N/A'}</td></tr>
<tr><td>Max Error</td><td class="metric">{f'{trk.max_error_px:.2f}px' if trk.max_error_px is not None else 'N/A'}</td></tr>
<tr><td>Within Threshold</td><td class="metric">{f'{trk.within_threshold_percent:.1f}%' if trk.within_threshold_percent is not None else 'N/A'}</td></tr>
</table>

<h2>Loss & Re-acquisition</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Loss Rate</td><td class="metric">{f'{los.loss_rate_percent:.1f}%' if los.loss_rate_percent is not None else 'N/A'}</td></tr>
<tr><td>Lock Retention</td><td class="metric">{f'{los.lock_retention_percent:.1f}%' if los.lock_retention_percent is not None else 'N/A'}</td></tr>
<tr><td>Re-acquisition Time</td><td class="metric">{f'{rea.mean_s:.3f}s' if rea.mean_s is not None else 'N/A'}</td></tr>
</table>

<h2>Processing Performance</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Duration</td><td class="metric">{result.duration:.1f}s</td></tr>
<tr><td>Frame Count</td><td class="metric">{result.frame_count}</td></tr>
<tr><td>Processed FPS</td><td class="metric">{f'{perf.processing_fps:.1f}' if perf.processing_fps is not None else 'N/A'}</td></tr>
<tr><td>Avg Latency</td><td class="metric">{f'{result.latency.total.mean_ms:.2f}ms' if result.latency.total.mean_ms is not None else 'N/A'}</td></tr>
<tr><td>P95 Latency</td><td class="metric">{f'{result.latency.total.p95_ms:.2f}ms' if result.latency.total.p95_ms is not None else 'N/A'}</td></tr>
</table>

<h2>Threshold Results</h2>
<table>
<tr><th>Threshold</th><th>Verdict</th></tr>
"""
    for tr in result.threshold_results:
        verdict_class = "pass" if tr.verdict.value == "PASS" else "fail"
        html += f'<tr><td>{tr.threshold.name if hasattr(tr.threshold, "name") else str(tr.threshold)}</td><td class="{verdict_class}">{tr.verdict.value}</td></tr>\n'

    html += """</table>
</body></html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


def generate_text_summary(report: dict[str, Any]) -> str:
    """Generate a human-readable text summary from a report dict."""
    lines = []
    lines.append("=" * 60)
    lines.append("FSOC TRACKER — PERFORMANCE REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated: {report.get('generated_at', 'N/A')}")
    lines.append("")

    sim = report.get("simulation", {})
    lines.append("SIMULATION")
    lines.append(f"  Duration:        {sim.get('duration_s', 0):.1f} s")
    lines.append(f"  Total Frames:    {sim.get('total_frames', 0)}")
    lines.append(f"  Average FPS:     {sim.get('avg_fps', 0):.1f}")
    lines.append(f"  Min FPS:         {sim.get('min_fps', 0):.1f}")
    lines.append(f"  Avg Latency:     {sim.get('avg_processing_ms', 0):.2f} ms")
    lines.append("")

    track = report.get("tracking", {})
    lines.append("TRACKING PERFORMANCE")
    lines.append(f"  Final State:     {track.get('final_state', 'N/A')}")
    lines.append(f"  Avg Error:       {track.get('avg_error_px', 0):.2f} px")
    lines.append(f"  Max Error:       {track.get('max_error_px', 0):.2f} px")
    lines.append(f"  RMSE:            {track.get('rmse_px', 0):.2f} px")
    lines.append(f"  P95 Error:       {track.get('p95_error_px', 0):.2f} px")
    lines.append(f"  Lock Retention:  {track.get('lock_retention_pct', 0):.1f}%")
    lines.append("")

    sc = report.get("sih_scorecard", {})
    lines.append("SIH SCORECARD (PS Requirements)")
    lines.append(f"  Acquisition:     {sc.get('acquisition_s', 'N/A')} s  (threshold: ≤{sc.get('acquisition_threshold_s', 2)} s)  {'PASS' if sc.get('acquisition_pass') else 'FAIL' if sc.get('acquisition_s') is not None else 'N/A'}")
    lines.append(f"  RMSE:            {sc.get('rmse_px', 'N/A')} px  (threshold: ≤{sc.get('rmse_threshold_px', 10)} px)  {'PASS' if sc.get('rmse_pass') else 'FAIL'}")
    lines.append(f"  Loss Rate:       {sc.get('loss_percent', 'N/A')} %  (threshold: <{sc.get('loss_threshold_pct', 5)} %)  {'PASS' if sc.get('loss_pass') else 'FAIL' if sc.get('loss_percent') is not None else 'N/A'}")
    lines.append(f"  Re-acquisition:  {sc.get('reacq_s', 'N/A')} s  (threshold: ≤{sc.get('reacq_threshold_s', 1)} s)  {'PASS' if sc.get('reacq_pass') else 'FAIL' if sc.get('reacq_s') is not None else 'N/A'}")
    lines.append(f"  Processing FPS:  {sc.get('fps', 'N/A')} fps  (threshold: ≥{sc.get('fps_threshold', 20)} fps)  {'PASS' if sc.get('fps_pass') else 'FAIL'}")
    lines.append("")

    cam = report.get("camera", {})
    lines.append("CAMERA")
    lines.append(f"  FOV:             {cam.get('hfov_deg', 0)}° x {cam.get('vfov_deg', 0)}°")
    lines.append(f"  Resolution:      {cam.get('resolution', 'N/A')}")
    lines.append("")

    dist = report.get("disturbances", {})
    lines.append("DISTURBANCES")
    lines.append(f"  Enabled:         {dist.get('enabled', False)}")
    lines.append(f"  Profile:         {dist.get('profile', 'N/A')}")
    lines.append(f"  Noise:           {dist.get('noise_type', 'none')} (σ={dist.get('noise_sigma', 0)})")
    lines.append(f"  Fog:             {dist.get('fog', 0)}")
    lines.append(f"  Jitter:          {dist.get('jitter_px', 0)} px")
    lines.append("")

    link = report.get("optical_link", {})
    lines.append("OPTICAL LINK")
    lines.append(f"  Status:          {link.get('status', 'N/A')}")
    lines.append(f"  Range:           {link.get('range_m', 0):.0f} m")
    lines.append(f"  Alignment:       {link.get('beam_alignment_pct', 0):.1f}%")
    lines.append(f"  Quality:         {link.get('link_quality_pct', 0):.1f}%")
    lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
