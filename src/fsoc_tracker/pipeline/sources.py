"""Frame source adapters — unified input abstraction.

Each mode (SIMULATION, VIDEO, LIVE, DATASET) provides a FrameSource
that yields Frame objects with proper timestamps. The pipeline consumes
only FrameSource, never knowing the underlying input type.

Information boundary:
    Runtime Frames NEVER carry simulator ground truth. Attach an
    :class:`~fsoc_tracker.pipeline.eval.EvalSink` for offline scoring.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np

from fsoc_tracker.core.interfaces import FrameSource
from fsoc_tracker.core.models import ColorModel, Frame, SourceType
from fsoc_tracker.pipeline.eval import EvalSink


class VirtualSimulationSource(FrameSource):
    """Yields frames from the simulation engine.

    Manages SimulationEngine + VirtualCamera + VirtualSensorRenderer +
    DisturbancePipeline internally. Each read() steps the simulation,
    renders a frame, and returns a runtime Frame (image + timing only).

    Optional ``eval_sink`` receives ground truth for offline scoring.
    Extra positional arguments after ``dt`` are rejected so a seed
    cannot accidentally enable a ground-truth flag.
    """

    def __init__(
        self,
        sim_engine: Any,
        camera: Any,
        sensor: Any,
        disturbance: Any | None = None,
        dt: float = 1.0 / 30.0,
        *,
        eval_sink: EvalSink | None = None,
        ground_truth_in_metadata: bool | None = None,
    ) -> None:
        if ground_truth_in_metadata:
            raise TypeError(
                "ground_truth_in_metadata is removed; attach eval_sink=EvalSink() "
                "for offline scoring. Runtime Frames never carry ground truth."
            )
        self._engine = sim_engine
        self._camera = camera
        self._sensor = sensor
        self._disturbance = disturbance
        self._dt = dt
        self._frame_index: int = 0
        self._sim_time: float = 0.0
        self._opened = False
        self._eval_sink = eval_sink
        # Strapdown carry: previous engine platform pose, so per-frame
        # platform motion (Terminal A itself moving) shifts/turns the
        # camera before rendering. None until open() baselines it.
        self._prev_platform: Any | None = None

    def open(self) -> None:
        self._opened = True
        self.sync_platform_baseline()

    def is_open(self) -> bool:
        return self._opened

    @property
    def nominal_fps(self) -> float | None:
        return 1.0 / self._dt if self._dt > 0 else None

    def read(self) -> Frame | None:
        if not self._opened:
            return None

        self._engine.step(self._dt)
        self._carry_platform()
        world_state = self._engine.get_state()
        targets = world_state.get_active_targets()

        render_camera = self._camera
        disturbance_context = None
        if self._disturbance is not None and self._disturbance.config.enabled:
            from fsoc_tracker.disturbances.context import (
                CameraPoseContext,
                DisturbanceContext,
            )

            disturbance_context = DisturbanceContext(
                timestamp_s=self._sim_time,
                dt=self._dt,
                frame_index=self._frame_index,
                image_width=self._camera.state.width,
                image_height=self._camera.state.height,
                camera_pose=CameraPoseContext(
                    position_x=self._camera.state.position_x,
                    position_y=self._camera.state.position_y,
                    position_z=self._camera.state.position_z,
                    pan_deg=self._camera.state.pan_deg,
                    tilt_deg=self._camera.state.tilt_deg,
                    roll_deg=self._camera.state.roll_deg,
                ),
            )
            effective_pose = self._disturbance.compute_effective_pose(
                disturbance_context.camera_pose,
                disturbance_context,
            )
            effective_state = replace(
                self._camera.state,
                position_x=effective_pose.position_x,
                position_y=effective_pose.position_y,
                position_z=effective_pose.position_z,
                pan_deg=effective_pose.pan_deg,
                tilt_deg=effective_pose.tilt_deg,
                roll_deg=effective_pose.roll_deg,
            )
            from fsoc_tracker.simulation.camera.camera import VirtualCamera
            render_camera = VirtualCamera(effective_state)

        # Source-layer disturbance: target disappearance
        if self._disturbance is not None and self._disturbance.should_suppress_target(self._sim_time):
            targets = []

        rendered = self._sensor.render(
            render_camera, targets, self._sim_time, self._frame_index,
        )

        image = rendered.image.copy()
        if disturbance_context is not None:
            image = self._disturbance.apply_to_image(image, disturbance_context)

        metadata: dict[str, Any] = {}
        if self._eval_sink is not None:
            primary_id = getattr(self._engine, "primary_beacon_id", None)
            if primary_id is None and rendered.ground_truths:
                # Single-target worlds never called set_primary_beacon.
                if len(rendered.ground_truths) == 1:
                    primary_id = rendered.ground_truths[0].target_id
            self._eval_sink.record(
                frame_index=self._frame_index,
                timestamp_s=self._sim_time,
                ground_truths=list(rendered.ground_truths),
                primary_id=primary_id,
            )

        h, w = image.shape[:2]
        ch = 1 if image.ndim == 2 else image.shape[2]
        color = ColorModel.GRAY if image.ndim == 2 else ColorModel.BGR

        frame = Frame(
            image=image,
            width=w,
            height=h,
            channels=ch,
            color_model=color,
            source_id="simulation",
            source_type=SourceType.SIMULATION,
            frame_index=self._frame_index,
            timestamp_s=self._sim_time,
            nominal_fps=1.0 / self._dt if self._dt > 0 else None,
            metadata=metadata,
        )

        self._sim_time += self._dt
        self._frame_index += 1
        return frame

    def sync_platform_baseline(self) -> None:
        """Re-baseline strapdown carry (call after manual placement).

        Manual place-terminal actions teleport the platform AND the
        camera together; without re-baselining, the next frame would
        apply the teleport as a motion delta a second time.
        """
        try:
            self._prev_platform = self._engine.get_terminal_a_state()
        except Exception:
            self._prev_platform = None

    def _carry_platform(self) -> None:
        """Move the mount with its platform (strapdown, in place)."""
        if self._camera is None:
            return
        try:
            cur = self._engine.get_terminal_a_state()
        except Exception:
            return
        if self._prev_platform is None:
            self._prev_platform = cur
            return
        try:
            from fsoc_tracker.simulation.platform import (
                apply_platform_delta_to_camera,
            )
            apply_platform_delta_to_camera(
                self._camera.state, self._prev_platform, cur)
        except Exception:
            pass
        self._prev_platform = cur

    def release(self) -> None:
        self._opened = False

    def get_info(self) -> dict[str, Any]:
        return {
            "source_type": "simulation",
            "width": self._camera.state.width if self._camera else 640,
            "height": self._camera.state.height if self._camera else 480,
            "nominal_fps": 1.0 / self._dt if self._dt > 0 else 30.0,
            "dt": self._dt,
        }

    @property
    def source_id(self) -> str:
        return "simulation"

    @property
    def camera(self) -> Any:
        """Access the virtual camera for control commands (actuator only)."""
        return self._camera

    @property
    def eval_sink(self) -> EvalSink | None:
        return self._eval_sink

    def set_eval_sink(self, sink: EvalSink | None) -> None:
        self._eval_sink = sink


class VideoSource(FrameSource):
    """Yields frames from an MP4/video file via OpenCV.

    Supports:
    - Standard perspective video (MP4, AVI, MOV, MKV, WebM)
    - Equirectangular / 360-degree video detection
    - Seek to arbitrary frame positions
    - Robust timestamp extraction from decoder or FPS fallback
    - Frame count and duration metadata
    """

    def __init__(self, video_path: str) -> None:
        self._path = video_path
        self._cap: Any = None
        self._frame_index: int = 0
        self._fps: float | None = None
        self._width: int = 0
        self._height: int = 0
        self._total_frames: int = 0
        self._duration_s: float = 0.0
        self._last_timestamp_s: float = -1.0
        self._is_equirectangular: bool = False

    def open(self) -> None:
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV required for video source: pip install opencv-python") from None
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self._path}")
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._fps = fps if fps > 0 else None
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self._fps and self._fps > 0 and self._total_frames > 0:
            self._duration_s = self._total_frames / self._fps
        self._last_timestamp_s = -1.0
        self._frame_index = 0
        self._detect_equirectangular()

    def _detect_equirectangular(self) -> None:
        """Heuristic: 360/equirectangular videos typically have 2:1 aspect ratio."""
        if self._width > 0 and self._height > 0:
            aspect = self._width / self._height
            self._is_equirectangular = 1.8 <= aspect <= 2.2

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> Frame | None:
        if self._cap is None:
            return None
        import cv2
        ret, bgr = self._cap.read()
        if not ret or bgr is None:
            return None

        position_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC))
        decoder_ts = position_ms / 1000.0
        if decoder_ts >= 0 and decoder_ts > self._last_timestamp_s:
            ts = decoder_ts
            timestamp_source = "decoder_position"
        elif self._fps is not None:
            ts = self._frame_index / self._fps
            timestamp_source = "nominal_fps"
        else:
            ts = float(self._frame_index)
            timestamp_source = "frame_index_unknown_rate"
        self._last_timestamp_s = ts
        h, w = bgr.shape[:2]
        frame = Frame(
            image=bgr,
            width=w,
            height=h,
            channels=3,
            color_model=ColorModel.BGR,
            source_id=self._path,
            source_type=SourceType.VIDEO,
            frame_index=self._frame_index,
            timestamp_s=ts,
            nominal_fps=self._fps,
            metadata={
                "video_path": self._path,
                "timestamp_source": timestamp_source,
                "is_equirectangular": self._is_equirectangular,
                "total_frames": self._total_frames,
                "duration_s": self._duration_s,
            },
        )
        self._frame_index += 1
        return frame

    def seek(self, frame_index: int) -> bool:
        """Seek to a specific frame number. Returns True on success."""
        if self._cap is None:
            return False
        import cv2
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        actual = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        if actual == frame_index:
            self._frame_index = frame_index
            self._last_timestamp_s = -1.0
            return True
        return False

    @property
    def position_frames(self) -> int | None:
        """Decoder's current frame position, or None when unavailable."""
        if self._cap is None:
            return None
        try:
            import cv2
            return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        except Exception:
            return None

    def reopen_and_seek(self, frame_index: int) -> bool:
        """Recover a stalled decoder: re-open the file and seek.

        OpenCV's ffmpeg backend goes sticky-EOF after a damaged packet:
        ``read()`` returns ``None`` forever even though later frames are
        perfectly decodable. Re-opening and seeking past the damage
        resumes the stream. Returns True when the seek landed.
        """
        try:
            self.release()
            self.open()
        except Exception:
            return False
        return self.seek(int(frame_index))

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_info(self) -> dict[str, Any]:
        return {
            "source_type": "video",
            "path": self._path,
            "width": self._width,
            "height": self._height,
            "fps": self._fps,
            "total_frames": self._total_frames,
            "duration_s": self._duration_s,
            "is_equirectangular": self._is_equirectangular,
        }

    @property
    def source_id(self) -> str:
        return self._path

    @property
    def nominal_fps(self) -> float | None:
        return self._fps

    @property
    def frame_count(self) -> int | None:
        return self._total_frames if self._total_frames > 0 else None

    @property
    def duration_s(self) -> float | None:
        return self._duration_s if self._duration_s > 0 else None

    @property
    def width(self) -> int | None:
        return self._width if self._width > 0 else None

    @property
    def height(self) -> int | None:
        return self._height if self._height > 0 else None

    @property
    def is_equirectangular(self) -> bool:
        return self._is_equirectangular


class LiveSource(FrameSource):
    """Yields frames from a live camera via OpenCV.

    Features:
    - Camera selection by ID
    - Resolution and FPS reporting
    - FPS tracking from frame timestamps
    - Camera availability detection
    - Graceful error handling
    """

    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480) -> None:
        self._camera_id = camera_id
        self._cap: Any = None
        self._frame_index: int = 0
        self._start_time: float = 0.0
        self._width: int = width
        self._height: int = height
        self._actual_fps: float = 0.0
        self._last_frame_time: float = 0.0
        self._error_message: str = ""

    def open(self) -> None:
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV required for live source: pip install opencv-python") from None
        self._cap = cv2.VideoCapture(self._camera_id)
        if not self._cap.isOpened():
            self._error_message = f"Cannot open camera {self._camera_id}"
            raise RuntimeError(self._error_message)
        self._start_time = time.monotonic()
        self._last_frame_time = self._start_time
        # Get actual resolution from camera
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def nominal_fps(self) -> float | None:
        return None

    @property
    def actual_fps(self) -> float:
        return self._actual_fps

    @property
    def width(self) -> int | None:
        return self._width if self._width > 0 else None

    @property
    def height(self) -> int | None:
        return self._height if self._height > 0 else None

    @property
    def error_message(self) -> str:
        return self._error_message

    def read(self) -> Frame | None:
        if self._cap is None:
            return None
        ret, bgr = self._cap.read()
        if not ret or bgr is None:
            return None

        current_time = time.monotonic()
        ts = current_time - self._start_time
        
        # Calculate FPS from frame interval
        if self._last_frame_time > 0:
            dt = current_time - self._last_frame_time
            if dt > 0:
                self._actual_fps = 0.9 * self._actual_fps + 0.1 * (1.0 / dt)
        self._last_frame_time = current_time

        h, w = bgr.shape[:2]
        frame = Frame(
            image=bgr,
            width=w,
            height=h,
            channels=3,
            color_model=ColorModel.BGR,
            source_id=f"camera_{self._camera_id}",
            source_type=SourceType.LIVE,
            frame_index=self._frame_index,
            timestamp_s=ts,
            metadata={
                "camera_id": self._camera_id,
                "actual_fps": self._actual_fps,
                "width": w,
                "height": h,
            },
        )
        self._frame_index += 1
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_info(self) -> dict[str, Any]:
        return {
            "source_type": "live",
            "camera_id": self._camera_id,
            "width": self._width,
            "height": self._height,
            "actual_fps": self._actual_fps,
            "is_available": self.is_open(),
            "error_message": self._error_message,
        }

    @property
    def source_id(self) -> str:
        return f"camera_{self._camera_id}"

    @staticmethod
    def enumerate_cameras(max_devices: int = 10) -> list[dict[str, Any]]:
        """Enumerate available camera devices.
        
        Returns list of dicts with camera_id, name, is_available.
        """
        import cv2
        cameras = []
        for i in range(max_devices):
            try:
                cap = cv2.VideoCapture(i)
                is_available = cap.isOpened()
                name = f"Camera {i}"
                if is_available:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    name = f"Camera {i} ({w}x{h})"
                    cap.release()
                cameras.append({
                    "camera_id": i,
                    "name": name,
                    "is_available": is_available,
                })
            except Exception:
                cameras.append({
                    "camera_id": i,
                    "name": f"Camera {i}",
                    "is_available": False,
                })
        return cameras

    @staticmethod
    def is_camera_available(camera_id: int) -> bool:
        """Check if a specific camera is available."""
        import cv2
        try:
            cap = cv2.VideoCapture(camera_id)
            is_available = cap.isOpened()
            cap.release()
            return is_available
        except Exception:
            return False


class DatasetSource(FrameSource):
    """Yields frames from a directory of images + optional ground truth JSONL.

    Labels go to ``eval_sink``, never onto the runtime Frame.
    """

    def __init__(
        self,
        image_dir: str,
        gt_path: str | None = None,
        *,
        eval_sink: EvalSink | None = None,
    ) -> None:
        self._dir = image_dir
        self._gt_path = gt_path
        self._files: list[str] = []
        self._frame_index: int = 0
        self._gt_data: list[dict] = []
        self._start_time: float = 0.0
        self._eval_sink = eval_sink

    def open(self) -> None:
        from pathlib import Path
        p = Path(self._dir)
        self._files = sorted(
            str(f) for f in p.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        )
        if not self._files:
            raise RuntimeError(f"No images found in {self._dir}")

        if self._gt_path:
            import json
            gt_path = Path(self._gt_path)
            if gt_path.exists():
                with open(gt_path) as f:
                    self._gt_data = [json.loads(line) for line in f if line.strip()]

        self._start_time = time.monotonic()

    def is_open(self) -> bool:
        return bool(self._files) and self._frame_index < len(self._files)

    def read(self) -> Frame | None:
        if self._frame_index >= len(self._files):
            return None

        import cv2
        path = self._files[self._frame_index]
        bgr = cv2.imread(path)
        if bgr is None:
            return None

        ts = time.monotonic() - self._start_time
        h, w = bgr.shape[:2]
        metadata: dict[str, Any] = {"image_path": path}

        if self._eval_sink is not None and self._frame_index < len(self._gt_data):
            row = self._gt_data[self._frame_index]
            self._eval_sink.record(
                frame_index=self._frame_index,
                timestamp_s=ts,
                ground_truths=[row],
                primary_id=row.get("target_id", 0) if isinstance(row, dict) else 0,
            )

        frame = Frame(
            image=bgr,
            width=w,
            height=h,
            channels=3,
            color_model=ColorModel.BGR,
            source_id=self._dir,
            source_type=SourceType.DATASET,
            frame_index=self._frame_index,
            timestamp_s=ts,
            metadata=metadata,
        )
        self._frame_index += 1
        return frame

    def release(self) -> None:
        self._files = []

    def get_info(self) -> dict[str, Any]:
        return {
            "source_type": "dataset",
            "directory": self._dir,
            "frame_count": len(self._files),
        }

    @property
    def source_id(self) -> str:
        return self._dir


# Backward-compatible alias
SimulationSource = VirtualSimulationSource


class EquirectangularFrameSource(FrameSource):
    """Wraps a VideoSource containing equirectangular/360 video.

    Extracts a perspective viewport from the equirectangular frame
    based on the virtual camera's current pan/tilt orientation.

    Pipeline:
        360 equirectangular frame
        → virtual camera orientation (pan, tilt)
        → perspective viewport extraction
        → perspective image for tracking

    The viewport is extracted by mapping pixel coordinates in the
    output image to spherical coordinates (longitude, latitude) on
    the equirectangular sphere, then sampling the source.

    Does not invent distance — monocular depth is N/A.
    """

    def __init__(
        self,
        video_source: VideoSource,
        camera: Any,
        output_width: int = 640,
        output_height: int = 480,
        hfov_deg: float = 4.0,
        vfov_deg: float = 3.0,
    ) -> None:
        self._source = video_source
        self._camera = camera
        self._out_w = output_width
        self._out_h = output_height
        self._hfov_rad = float(np.deg2rad(hfov_deg))
        self._vfov_rad = float(np.deg2rad(vfov_deg))
        self._opened = False

    def open(self) -> None:
        self._source.open()
        self._opened = True

    def is_open(self) -> bool:
        return self._opened and self._source.is_open()

    def read(self) -> Frame | None:
        raw = self._source.read()
        if raw is None:
            return None

        if self._camera is None:
            return raw

        pan_deg = self._camera.state.pan_deg
        tilt_deg = self._camera.state.tilt_deg

        viewport = self._extract_viewport(raw.image, pan_deg, tilt_deg)
        h, w = viewport.shape[:2]
        ch = 1 if viewport.ndim == 2 else viewport.shape[2]
        color = ColorModel.GRAY if viewport.ndim == 2 else ColorModel.BGR

        return Frame(
            image=viewport,
            width=w,
            height=h,
            channels=ch,
            color_model=color,
            source_id=raw.source_id,
            source_type=raw.source_type,
            frame_index=raw.frame_index,
            timestamp_s=raw.timestamp_s,
            nominal_fps=raw.nominal_fps,
            monotonic_time_s=raw.monotonic_time_s,
            metadata={
                **raw.metadata,
                "equirectangular_viewport": True,
                "viewport_pan_deg": pan_deg,
                "viewport_tilt_deg": tilt_deg,
                "viewport_hfov_deg": float(np.rad2deg(self._hfov_rad)),
                "viewport_vfov_deg": float(np.rad2deg(self._vfov_rad)),
                "source_is_equirectangular": True,
            },
        )

    def _extract_viewport(
        self, equirect: np.ndarray, pan_deg: float, tilt_deg: float,
    ) -> np.ndarray:
        """Extract a perspective viewport from an equirectangular frame.

        Maps each output pixel to a (longitude, latitude) on the sphere,
        then samples the equirectangular image.

        Args:
            equirect: Equirectangular source image (H, W, 3) BGR.
            pan_deg: Camera horizontal angle (longitude center) in degrees.
            tilt_deg: Camera vertical angle (latitude center) in degrees.
            Negative tilt = looking up, positive = looking down.

        Returns:
            Perspective viewport image (out_h, out_w, 3) BGR.
        """
        src_h, src_w = equirect.shape[:2]
        pan_rad = float(np.deg2rad(pan_deg))
        tilt_rad = float(np.deg2rad(tilt_deg))

        half_hfov = self._hfov_rad / 2.0
        half_vfov = self._vfov_rad / 2.0

        # Generate a grid of (u, v) in [-1, 1] for the output viewport
        u = np.linspace(-1, 1, self._out_w, dtype=np.float64)
        v = np.linspace(-1, 1, self._out_h, dtype=np.float64)
        uu, vv = np.meshgrid(u, v)

        # Map to spherical longitude/latitude offsets
        lon_offset = uu * half_hfov
        lat_offset = -vv * half_vfov  # negative: top of image = up

        # Absolute spherical coords
        longitude = pan_rad + lon_offset
        latitude = tilt_rad + lat_offset

        # Clamp latitude to [-pi/2, pi/2]
        latitude = np.clip(latitude, -np.pi / 2, np.pi / 2)

        # Map to equirectangular pixel coords
        # longitude [-pi, pi] → x [0, src_w]
        # latitude [pi/2, -pi/2] → y [0, src_h]
        px_x = ((longitude + np.pi) / (2.0 * np.pi)) * src_w
        px_y = ((np.pi / 2.0 - latitude) / np.pi) * src_h

        # Wrap x for longitude (seamless)
        px_x = np.mod(px_x, src_w)

        # Clamp y
        px_y = np.clip(px_y, 0, src_h - 1)

        # Sample with nearest-neighbor (fast, no interpolation artifacts)
        map_x = px_x.astype(np.float32)
        map_y = px_y.astype(np.float32)

        try:
            import cv2
            viewport = cv2.remap(
                equirect, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )
        except Exception:
            # Fallback: manual nearest-neighbor sampling
            ix = np.round(px_x).astype(np.intp) % src_w
            iy = np.round(px_y).astype(np.intp)
            iy = np.clip(iy, 0, src_h - 1)
            viewport = equirect[iy, ix]

        return viewport

    def release(self) -> None:
        self._source.release()
        self._opened = False

    def get_info(self) -> dict[str, Any]:
        info = self._source.get_info()
        info["equirectangular_viewport"] = True
        info["output_width"] = self._out_w
        info["output_height"] = self._out_h
        info["viewport_hfov_deg"] = float(np.rad2deg(self._hfov_rad))
        info["viewport_vfov_deg"] = float(np.rad2deg(self._vfov_rad))
        return info

    @property
    def source_id(self) -> str:
        return f"equirect:{self._source.source_id}"

    @property
    def nominal_fps(self) -> float | None:
        return self._source.nominal_fps

    @property
    def frame_count(self) -> int | None:
        return self._source.frame_count

    @property
    def duration_s(self) -> float | None:
        return self._source.duration_s

    @property
    def width(self) -> int | None:
        return self._out_w

    @property
    def height(self) -> int | None:
        return self._out_h

    def seek(self, frame_index: int) -> bool:
        return self._source.seek(frame_index)

    @property
    def position_frames(self) -> int | None:
        """Decoder position of the wrapped video source, if known."""
        getter = getattr(self._source, "position_frames", None)
        try:
            return int(getter()) if callable(getter) else (
                int(getter) if getter is not None else None)
        except Exception:
            return None

    def reopen_and_seek(self, frame_index: int) -> bool:
        """Recover a stalled decoder via the wrapped video source."""
        inner = getattr(self._source, "reopen_and_seek", None)
        if not callable(inner):
            return False
        ok = bool(inner(int(frame_index)))
        if ok:
            self._opened = True
        return ok

    @property
    def is_equirectangular(self) -> bool:
        return True


class LiveViewportSource(FrameSource):
    """Software virtual viewport (digital PTZ) over a fixed camera feed.

    Used when no physical PTZ hardware is available: a fixed-size window
    follows the commanded pan/tilt orientation with the same FOV semantics
    as the virtual camera (``output_width`` pixels span ``hfov_deg``).
    No depth is invented; timestamps and frame indices pass through
    untouched from the underlying source.
    """

    def __init__(
        self,
        source: FrameSource,
        output_width: int = 640,
        output_height: int = 480,
        hfov_deg: float = 4.0,
        vfov_deg: float = 3.0,
    ) -> None:
        from fsoc_tracker.simulation.camera.camera import VirtualCamera
        from fsoc_tracker.simulation.camera.state import CameraState

        self._source = source
        self._out_w = max(1, int(output_width))
        self._out_h = max(1, int(output_height))
        self._hfov = max(0.1, float(hfov_deg))
        self._vfov = max(0.1, float(vfov_deg))
        self._camera = VirtualCamera(CameraState(
            width=self._out_w,
            height=self._out_h,
            horizontal_fov_deg=self._hfov,
            vertical_fov_deg=self._vfov,
        ))
        self._pan_deg = 0.0
        self._tilt_deg = 0.0

    @property
    def camera(self) -> Any:
        """Software PTZ camera used by the runtime actuator."""
        return self._camera

    def set_viewport_center(self, pan_deg: float, tilt_deg: float) -> None:
        """Aim the viewport window (degrees, same convention as camera)."""
        self._pan_deg = float(pan_deg)
        self._tilt_deg = float(tilt_deg)
        self._camera.state.pan_deg = self._pan_deg
        self._camera.state.tilt_deg = self._tilt_deg

    def open(self) -> None:
        self._source.open()

    def is_open(self) -> bool:
        try:
            return bool(self._source.is_open())
        except Exception:
            return False

    def read(self) -> Frame | None:
        raw = self._source.read()
        if raw is None:
            return None
        self._pan_deg = self._camera.state.pan_deg
        self._tilt_deg = self._camera.state.tilt_deg
        img = raw.image
        h, w = img.shape[:2]
        px_per_deg_x = self._out_w / self._hfov
        px_per_deg_y = self._out_h / self._vfov
        cx = w / 2.0 + self._pan_deg * px_per_deg_x
        cy = h / 2.0 + self._tilt_deg * px_per_deg_y
        win_w = min(self._out_w, w)
        win_h = min(self._out_h, h)
        x1 = int(min(max(cx - win_w / 2.0, 0.0), max(w - win_w, 0.0)))
        y1 = int(min(max(cy - win_h / 2.0, 0.0), max(h - win_h, 0.0)))
        crop = img[y1:y1 + win_h, x1:x1 + win_w]
        if (win_w, win_h) != (self._out_w, self._out_h):
            try:
                import cv2
                crop = cv2.resize(crop, (self._out_w, self._out_h),
                                  interpolation=cv2.INTER_LINEAR)
            except Exception:
                pass
        oh, ow = crop.shape[:2]
        ch = 1 if crop.ndim == 2 else crop.shape[2]
        color = ColorModel.GRAY if crop.ndim == 2 else raw.color_model
        metadata = dict(raw.metadata)
        metadata.update({
            "viewport_active": True,
            "viewport_pan_deg": self._pan_deg,
            "viewport_tilt_deg": self._tilt_deg,
            "viewport_window": [x1, y1, win_w, win_h],
        })
        return Frame(
            image=crop,
            width=ow,
            height=oh,
            channels=ch,
            color_model=color,
            source_id=raw.source_id,
            source_type=raw.source_type,
            frame_index=raw.frame_index,
            timestamp_s=raw.timestamp_s,
            nominal_fps=raw.nominal_fps,
            monotonic_time_s=raw.monotonic_time_s,
            metadata=metadata,
        )

    def release(self) -> None:
        self._source.release()

    def get_info(self) -> dict[str, Any]:
        info = self._source.get_info() if hasattr(self._source, "get_info") else {}
        info = dict(info)
        info["viewport_active"] = True
        info["output_width"] = self._out_w
        info["output_height"] = self._out_h
        return info

    @property
    def source_id(self) -> str:
        return f"viewport:{self._source.source_id}"

    @property
    def nominal_fps(self) -> float | None:
        return self._source.nominal_fps

    @property
    def width(self) -> int | None:
        return self._out_w

    @property
    def height(self) -> int | None:
        return self._out_h
