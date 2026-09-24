"""Long-video endurance: no early stops, decoder recovery, bounded memory.

Regression cover for the "5-minute video stops early" failure mode:
OpenCV's ffmpeg backend goes sticky-EOF after a damaged packet —
read() returns None forever even though later frames are decodable —
and the worker used to stop the run after 30 such misses. The worker
now re-opens + seeks past the damage and stops only on confirmed EOF.

Also covers the 24-hour requirement: per-frame accumulators must not
grow without bound (PipelineState uses array('d'); tracker events are
a capped ring).
"""

from __future__ import annotations

from array import array

import cv2
import numpy as np
import pytest

from fsoc_tracker.core.interfaces import FrameSource
from fsoc_tracker.core.models import ColorModel, Frame, SourceType
from fsoc_tracker.pipeline.sources import VideoSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _beacon_frame(width: int, height: int, cx: int, cy: int) -> np.ndarray:
    img = np.full((height, width, 3), 20, dtype=np.uint8)
    img[cy - 2:cy + 3, cx - 2:cx + 3] = [230, 230, 230]
    return img


def _write_beacon_video(path: str, num_frames: int = 120,
                        width: int = 64, height: int = 48,
                        noisy: bool = False) -> str:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 30.0, (width, height))
    assert out.isOpened(), "VideoWriter failed to open"
    rng = np.random.default_rng(0)
    for i in range(num_frames):
        cx = int(width * 0.2 + (width * 0.6) * i / max(num_frames - 1, 1))
        cy = height // 2
        img = _beacon_frame(width, height, cx, cy)
        if noisy:
            # Per-frame noise inflates mdat so a mid-file corrupt chunk
            # damages some packets while leaving a recoverable tail.
            noise = rng.integers(0, 60, size=img.shape, dtype=np.uint8)
            img = cv2.add(img, noise)
        out.write(img)
    out.release()
    return path


def _corrupt_middle(path: str, size: int = 262144) -> None:
    """Damage bytes inside the mdat payload (never the moov index).

    Parses top-level boxes to find 'mdat', then flips a chunk in the
    middle of its media data — the corrupt-packet scenario real-world
    files hit (truncated downloads, bad sectors, flaky encoders).
    The default 256 KB span destroys dozens of packets so the decoder
    stalls long past the legacy 30-miss kill threshold.
    """
    import struct
    with open(path, "r+b") as f:
        data = f.read()
    off = 0
    mdat_at = mdat_len = 0
    while off + 8 <= len(data):
        (size32, typ) = struct.unpack(">I4s", data[off:off + 8])
        typ = typ.decode("ascii", "replace")
        if size32 == 1:  # largesize
            (size64,) = struct.unpack(">Q", data[off + 8:off + 16])
            hlen, blen = 16, size64
        elif size32 == 0:  # extends to EOF
            hlen, blen = 8, len(data) - off
        else:
            hlen, blen = 8, size32
        if blen <= 0:
            break
        if typ == "mdat" and blen > hlen + size * 2:
            mdat_at, mdat_len = off + hlen, blen - hlen
            break
        off += blen
    assert mdat_at > 0, "no mdat box found to corrupt"
    with open(path, "r+b") as f:
        f.seek(mdat_at + mdat_len // 2)
        f.write(bytes([0xFF, 0x00, 0xFF, 0x00] * (size // 4)))


@pytest.fixture
def qapp():
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class _ScriptedVideo(FrameSource):
    """Video-like source with scripted sticky-failure + recovery support."""

    def __init__(self, total: int = 200, fail_from: int = 10,
                 fail_until: int = 60) -> None:
        self._n = -1
        self._total = total
        self._fail_from = fail_from
        self._fail_until = fail_until
        self._recovered = False
        self.reopen_calls: list[int] = []

    def open(self) -> None:
        pass

    def is_open(self) -> bool:
        return True

    def release(self) -> None:
        pass

    @property
    def source_id(self) -> str:
        return "scripted-video"

    @property
    def frame_count(self) -> int:
        return self._total

    @property
    def position_frames(self) -> int:
        return self._n + 1

    def reopen_and_seek(self, frame_index: int) -> bool:
        self.reopen_calls.append(int(frame_index))
        self._n = int(frame_index) - 1
        self._recovered = True
        return True

    def read(self) -> Frame | None:
        nxt = self._n + 1
        if nxt >= self._total:
            return None
        if (self._fail_from <= nxt < self._fail_until
                and not self._recovered):
            return None  # sticky until recovery, like a damaged packet
        self._n = nxt
        img = _beacon_frame(640, 480, 320, 240)
        return Frame(image=img.copy(), width=640, height=480,
                     channels=3, color_model=ColorModel.BGR,
                     source_id="scripted-video",
                     source_type=SourceType.VIDEO,
                     frame_index=self._n,
                     timestamp_s=self._n / 30.0, nominal_fps=30.0)


def _video_worker(qapp, source: FrameSource,
                  real_video_path: str) -> tuple:
    from fsoc_tracker.gui.worker import ProcessingWorker
    msgs: list[str] = []
    w = ProcessingWorker()
    w.log.connect(lambda level, msg: msgs.append(f"{level}: {msg}"))
    w.configure({"mode": "video", "video_path": real_video_path})
    w._init_pipeline()
    w._source = source
    w._source.open()
    w._ensure_tracking_pipeline()
    w._running = True
    return w, msgs


# ---------------------------------------------------------------------------
# 1. Corrupt-packet mechanism: sticky None, recovery via reopen+seek
# ---------------------------------------------------------------------------

class TestCorruptPacketRecovery:
    def test_sticky_eof_then_reopen_recovers(self, tmp_path):
        clean = _write_beacon_video(str(tmp_path / "v.mp4"), num_frames=300,
                                    width=320, height=240, noisy=True)
        # 1 MB span: destroys enough packets to stall ~50 consecutive
        # reads (measured), past the legacy 30-miss kill threshold.
        _corrupt_middle(clean, size=1048576)
        vs = VideoSource(clean)
        vs.open()
        total = vs.frame_count
        assert total == 300

        good = 0
        while True:
            f = vs.read()
            if f is None:
                break
            good += 1
            assert good <= total + 5  # safety bound

        extra_nones = sum(1 for _ in range(40) if vs.read() is None)
        vs.release()
        # The stall must exceed the legacy 30-miss kill threshold:
        # the old code would have stopped the run right here.
        if not (0 < good < total and extra_nones >= 30):
            pytest.skip("codec tolerated the injected damage "
                        "(no stall to recover from)")

        # Seek past the damage (scanning forward like the worker's
        # growing-skip recovery): later frames are perfectly decodable.
        vs2 = VideoSource(clean)
        vs2.open()
        landed = None
        for target in range(good + 10, total, 20):
            if vs2.reopen_and_seek(target):
                f = vs2.read()
                if f is not None and f.frame_index == target:
                    landed = target
                    break
        assert landed is not None, "no recoverable frame found past damage"
        recovered = 1
        while True:
            f = vs2.read()
            if f is None:
                break
            recovered += 1
        vs2.release()
        assert recovered > 0
        assert landed + recovered >= total - 20


# ---------------------------------------------------------------------------
# 2. Worker keeps a long run alive across a decoder stall
# ---------------------------------------------------------------------------

class TestWorkerDecoderRecovery:
    def test_stall_recovers_and_run_continues(self, qapp, tmp_path):
        real = _write_beacon_video(str(tmp_path / "real.mp4"),
                                   num_frames=30)
        src = _ScriptedVideo(total=200, fail_from=10, fail_until=60)
        w, msgs = _video_worker(qapp, src, real)
        steps = 120
        for _ in range(steps):
            w._run_one_step()
            qapp.processEvents()
        try:
            assert w._running is True
            assert w._video_miss_streak == 0
            # Only the 4 pre-recovery misses are lost (5th triggers
            # recovery inside the same step and processes a frame).
            assert w._frames_processed == steps - 4
            assert len(src.reopen_calls) >= 1
            assert any("recovered" in m for m in msgs)
        finally:
            w.stop_run()

    def test_true_eof_stops_with_frame_counts(self, qapp, tmp_path):
        real = _write_beacon_video(str(tmp_path / "real.mp4"),
                                   num_frames=30)
        src = _ScriptedVideo(total=50, fail_from=10 ** 9,
                             fail_until=10 ** 9)
        w, msgs = _video_worker(qapp, src, real)
        for _ in range(200):
            w._run_one_step()
            qapp.processEvents()
            if not w._running:
                break
        try:
            assert w._running is False
            assert w._frames_processed == 50
            assert any("Video ended after 50 frames (50 total)" in m
                       for m in msgs)
            assert len(src.reopen_calls) == 0  # clean EOF: no recovery needed
        finally:
            w.stop_run()


# ---------------------------------------------------------------------------
# 3. Memory stays bounded on very long runs
# ---------------------------------------------------------------------------

class TestBoundedAccumulators:
    def test_pipeline_state_arrays_and_tracker_ring(self):
        from fsoc_tracker.control.controller import CoarsePointingController
        from fsoc_tracker.perception.classical_engine import (
            ClassicalBeaconDetector,
        )
        from fsoc_tracker.perception.config import PerceptionConfig
        from fsoc_tracker.pipeline.pipeline import TrackingPipeline
        from fsoc_tracker.tracking.config import TrackerConfig
        from fsoc_tracker.tracking.state import TrackState
        from fsoc_tracker.tracking.tracker import KalmanTracker

        pipe = TrackingPipeline(
            perception=ClassicalBeaconDetector(PerceptionConfig.for_video()),
            tracker=KalmanTracker(TrackerConfig()),
            controller=CoarsePointingController(),
        )
        src = _ScriptedVideo(total=2000, fail_from=10 ** 9,
                             fail_until=10 ** 9)
        pipe.set_source(src)
        pipe.start()
        for _ in range(2000):
            assert pipe.step() is not None
        pipe.stop()
        ps = pipe.state
        assert ps.frame_count == 2000
        for name in ("errors", "errors_x", "errors_y",
                     "processing_times_ms", "perception_times_ms",
                     "tracking_times_ms", "control_times_ms"):
            arr = getattr(ps, name)
            assert isinstance(arr, array) and arr.itemsize == 8, name
        # 8 bytes/entry: 2000 frames x 7 histories ~ 0.1 MB, not ~0.5 MB boxed.

        # Force hundreds of tracker transitions: the event log must cap.
        from fsoc_tracker.perception.models import BeaconDetection
        trk = KalmanTracker(TrackerConfig())
        det = BeaconDetection(detected=True, confidence=0.9,
                              center_x=320.0, center_y=240.0)
        ts = 0.0
        for i in range(400):
            ts += 1.0 / 30.0
            trk._sm.force_state(TrackState.NO_TRACK, ts)
            trk.update([det], ts)
        assert len(trk.events) <= 512
        assert trk._events.maxlen == 512
        # Newest events survive the cap.
        assert trk.events[-1].timestamp_s == pytest.approx(ts)
