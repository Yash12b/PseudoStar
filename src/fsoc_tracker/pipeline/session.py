"""Session controller — lifecycle management for pipeline runs.

Provides start/pause/resume/stop/reset and captures run metadata,
state, final metrics, and output artifacts.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from fsoc_tracker.pipeline.pipeline import PipelineState, TrackingPipeline


class RunState(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


@dataclass
class RunMetadata:
    session_id: str = ""
    start_time: str = ""
    end_time: str = ""
    source_type: str = ""
    source_info: str = ""
    detector_name: str = ""
    tracker_name: str = ""
    controller_name: str = ""
    disturbance_profile: str = ""
    random_seed: int = 42
    configuration_hash: str = ""
    software_version: str = "0.1.0"


@dataclass
class RunResult:
    metadata: RunMetadata = field(default_factory=RunMetadata)
    state: RunState = RunState.IDLE
    pipeline_state: PipelineState | None = None
    duration_s: float = 0.0
    frame_count: int = 0
    errors: list[float] = field(default_factory=list)
    acquisition_time_s: float | None = None
    loss_events: int = 0
    reacquisition_times: list[float] = field(default_factory=list)
    processing_fps: float = 0.0
    mean_processing_ms: float = 0.0
    artifacts_dir: str = ""
    # Per-frame centroid log for external comparison (BP2 evaluators
    # compare these against predefined error values). Rows:
    # (frame_index, timestamp_s, detected, detection_x, detection_y,
    #  confidence, est_x, est_y, track_state, locked).
    centroids: list[tuple] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Session: {self.metadata.session_id}",
            f"State: {self.state.value}",
            f"Frames: {self.frame_count}",
            f"Duration: {self.duration_s:.2f}s",
            f"FPS: {self.processing_fps:.1f}",
            f"Mean latency: {self.mean_processing_ms:.1f}ms",
        ]
        if self.acquisition_time_s is not None:
            lines.append(f"Acquisition: {self.acquisition_time_s:.3f}s")
        if self.errors:
            import math
            rmse = math.sqrt(sum(e**2 for e in self.errors) / len(self.errors))
            lines.append(f"RMSE: {rmse:.2f}px")
            lines.append(f"Max error: {max(self.errors):.2f}px")
        lines.append(f"Loss events: {self.loss_events}")
        if self.reacquisition_times:
            lines.append(f"Reacquisition: {sum(self.reacquisition_times)/len(self.reacquisition_times):.3f}s (mean)")
        return "\n".join(lines)


class SessionController:
    """High-level lifecycle manager for pipeline runs.

    Usage::

        session = SessionController(pipeline)
        session.start()
        while session.state == RunState.RUNNING:
            result = session.step()
        final = session.stop()
        print(final.summary())
    """

    def __init__(
        self,
        pipeline: TrackingPipeline,
        output_dir: str = "runs",
        auto_save: bool = True,
    ) -> None:
        self._pipeline = pipeline
        self._output_dir = Path(output_dir)
        self._auto_save = auto_save
        self._state = RunState.IDLE
        self._metadata = RunMetadata()
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._centroid_rows: list[tuple] = []

    @property
    def state(self) -> RunState:
        return self._state

    @property
    def metadata(self) -> RunMetadata:
        return self._metadata

    @property
    def pipeline(self) -> TrackingPipeline:
        return self._pipeline

    def configure(
        self,
        source_type: str = "simulation",
        source_info: str = "",
        detector_name: str = "classical",
        tracker_name: str = "kalman",
        controller_name: str = "pid",
        disturbance_profile: str = "clear",
        random_seed: int = 42,
        config_dict: dict[str, Any] | None = None,
    ) -> None:
        """Configure session metadata."""
        self._metadata = RunMetadata(
            session_id=uuid.uuid4().hex[:12],
            start_time="",
            source_type=source_type,
            source_info=source_info,
            detector_name=detector_name,
            tracker_name=tracker_name,
            controller_name=controller_name,
            disturbance_profile=disturbance_profile,
            random_seed=random_seed,
            configuration_hash=self._hash_config(config_dict or {}),
        )

    def start(self) -> None:
        """Start the pipeline."""
        if self._state == RunState.RUNNING:
            return
        self._pipeline.start()
        self._state = RunState.RUNNING
        self._start_time = time.monotonic()
        self._metadata.start_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def pause(self) -> None:
        """Pause the pipeline."""
        if self._state == RunState.RUNNING:
            self._state = RunState.PAUSED

    def resume(self) -> None:
        """Resume from pause."""
        if self._state == RunState.PAUSED:
            self._state = RunState.RUNNING

    def step(self) -> Any:
        """Process one frame. Returns PipelineFrameResult or None."""
        if self._state != RunState.RUNNING:
            return None
        result = self._pipeline.step()
        if result is None:
            self._state = RunState.STOPPED
            return None
        self._centroid_rows.append(self._centroid_row(result))
        return result

    @staticmethod
    def _centroid_row(result: Any) -> tuple:
        """Extract the per-frame centroid comparison row."""
        perc = result.perception
        trk = result.tracking
        det = perc is not None and bool(perc.detected)
        prim = perc.primary_detection if perc is not None else None
        return (
            result.frame_index,
            round(result.timestamp_s, 4),
            int(det),
            round(float(prim.center_x), 2) if prim is not None else "",
            round(float(prim.center_y), 2) if prim is not None else "",
            round(float(prim.confidence), 3) if prim is not None else "",
            round(float(trk.estimated_x), 2) if trk is not None else "",
            round(float(trk.estimated_y), 2) if trk is not None else "",
            trk.state.name if trk is not None else "",
            int(bool(getattr(trk, "locked", False))) if trk is not None else 0,
        )

    def stop(self) -> RunResult:
        """Stop and produce final result."""
        self._end_time = time.monotonic()
        self._pipeline.stop()
        self._state = RunState.STOPPED
        self._metadata.end_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        result = self._build_result()

        if self._auto_save:
            self._save_artifacts(result)

        return result

    def reset(self) -> None:
        """Reset for a new run."""
        self._pipeline.reset()
        self._state = RunState.IDLE
        self._centroid_rows = []

    def _build_result(self) -> RunResult:
        ps = self._pipeline.state
        duration = self._end_time - self._start_time if self._end_time > 0 else 0.0
        fps = ps.frame_count / duration if duration > 0 else 0.0
        mean_ms = (
            sum(ps.processing_times_ms) / len(ps.processing_times_ms)
            if ps.processing_times_ms else 0.0
        )

        return RunResult(
            metadata=self._metadata,
            state=self._state,
            pipeline_state=ps,
            duration_s=duration,
            frame_count=ps.frame_count,
            errors=list(ps.errors),
            acquisition_time_s=ps.acquisition_time_s,
            loss_events=ps.loss_events,
            reacquisition_times=list(ps.reacquisition_times),
            processing_fps=fps,
            mean_processing_ms=mean_ms,
            centroids=list(self._centroid_rows),
        )

    def _save_artifacts(self, result: RunResult) -> None:
        """Save run artifacts to output directory."""
        session_dir = self._output_dir / result.metadata.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        result.artifacts_dir = str(session_dir)

        # Metadata
        meta = {
            "session_id": result.metadata.session_id,
            "start_time": result.metadata.start_time,
            "end_time": result.metadata.end_time,
            "source_type": result.metadata.source_type,
            "source_info": result.metadata.source_info,
            "detector": result.metadata.detector_name,
            "tracker": result.metadata.tracker_name,
            "controller": result.metadata.controller_name,
            "disturbance": result.metadata.disturbance_profile,
            "seed": result.metadata.random_seed,
            "config_hash": result.metadata.configuration_hash,
            "software_version": result.metadata.software_version,
            "duration_s": result.duration_s,
            "frame_count": result.frame_count,
            "processing_fps": result.processing_fps,
            "mean_processing_ms": result.mean_processing_ms,
            "acquisition_time_s": result.acquisition_time_s,
            "loss_events": result.loss_events,
        }
        with open(session_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Errors CSV
        if result.errors:
            with open(session_dir / "errors.csv", "w") as f:
                f.write("frame,error_px\n")
                for i, e in enumerate(result.errors):
                    f.write(f"{i},{e:.4f}\n")

        # Per-frame centroid log (BP2: compare against predefined values)
        with open(session_dir / "centroids.csv", "w") as f:
            f.write("frame_index,timestamp_s,detected,detection_x,"
                    "detection_y,confidence,est_x,est_y,track_state,"
                    "locked\n")
            for row in result.centroids:
                f.write(",".join(str(v) for v in row) + "\n")

        # Summary
        with open(session_dir / "summary.txt", "w") as f:
            f.write(result.summary())

    def _hash_config(self, config: dict[str, Any]) -> str:
        """Compute a stable hash of the configuration."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
