"""Recompute-verify reports + handoff-ready signal.

Covers verify_performance_report (PASS on honest reports, FAIL on
tampered aggregates or missing raw data) and the worker's GT-free
handoff-ready streak (stable TRACKING -> READY + log, instability
clears it).
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def qapp():
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _report_with_raw(tmp_path, errors: list[float]) -> str:
    from fsoc_tracker.benchmark.report import generate_performance_report

    class _E:
        timestamps = [float(i) / 30.0 for i in range(len(errors))]
        errors_euclidean = list(errors)
        fps_history = [30.0] * len(errors)

    class _P:
        frames_processed = len(errors)
        processing_ms = 3.0
        perception_ms = 2.0

    class _T:
        state = "TRACKING"
        locked = True

    class _S:
        acquisition_s = 0.1
        acquisition_threshold = 2.0
        rmse_threshold = 10.0
        loss_percent = 1.0
        loss_threshold = 5.0
        reacq_s = 0.05
        reacq_threshold = 1.0
        fps_threshold = 20.0
        lock_retention_pct = 99.0

    class _C:
        hfov_deg = 4.0
        vfov_deg = 3.0
        image_width = 640
        image_height = 480
        pan_deg = 0.0
        tilt_deg = 0.0

    class _D:
        enabled = False
        profile = "clear"
        noise_type = "none"
        noise_sigma = 0.0
        fog = 0.0
        haze = 0.0
        rain = 0.0
        jitter_px = 0.0

    class _L:
        status = "LOCKED"
        range_m = 100.0
        beam_alignment_percent = 99.0
        angular_error_deg = 0.01
        link_quality_percent = 99.0

    class _A:
        situation = "normal_tracking"
        action = "track"
        confidence = 0.9
        explanation = "ok"
        fallback_active = False
        search_active = False

    class _State:
        elapsed_s = 10.0
        errors = _E()
        performance = _P()
        tracking = _T()
        scorecard = _S()
        camera = _C()
        disturbances = _D()
        optical_link = _L()
        ai_state = _A()
        handoff_ready = True
        handoff_stable_s = 2.0

    return generate_performance_report(
        _State(), output_dir=str(tmp_path), filename="rep.json",
        raw_errors=list(errors))


class TestRecomputeVerify:
    def test_roundtrip_passes(self, tmp_path):
        from fsoc_tracker.benchmark.report import verify_performance_report
        path = _report_with_raw(tmp_path, [1.0, 2.0, 3.0] * 50)
        verdict = verify_performance_report(path)
        assert verdict["ok"] is True
        assert verdict["frames"] == 150
        assert all(d["match"] for d in verdict["details"].values())

    def test_tampered_aggregate_fails(self, tmp_path):
        from fsoc_tracker.benchmark.report import verify_performance_report
        path = _report_with_raw(tmp_path, [1.0, 2.0, 3.0] * 50)
        with open(path, encoding="utf-8") as f:
            rep = json.load(f)
        rep["tracking"]["rmse_px"] = 0.01  # tamper with the headline number
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rep, f)
        verdict = verify_performance_report(path)
        assert verdict["ok"] is False
        assert verdict["details"]["rmse_px"]["match"] is False

    def test_missing_raw_fails(self, tmp_path):
        from fsoc_tracker.benchmark.report import verify_performance_report
        path = _report_with_raw(tmp_path, [1.0, 2.0])
        with open(path, encoding="utf-8") as f:
            rep = json.load(f)
        import os
        os.remove(rep["raw_errors_path"])
        verdict = verify_performance_report(path)
        assert verdict["ok"] is False

    def test_cli_verify_passes(self, tmp_path):
        import subprocess
        import sys
        path = _report_with_raw(tmp_path, [0.5, 1.5] * 20)
        proc = subprocess.run(
            [sys.executable, "-m", "fsoc_tracker.cli.run_benchmark",
             "verify", path],
            capture_output=True, text=True, cwd="/Users/yashas/Downloads/fsoc_tracker",
            env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                 "PYTHONPATH": "/Users/yashas/Downloads/fsoc_tracker/src"},
        )
        assert proc.returncode == 0, proc.stderr[-500:]
        assert "VERIFY: PASS" in proc.stdout


class TestHandoffReady:
    def _worker(self, qapp):
        from fsoc_tracker.gui.worker import ProcessingWorker
        w = ProcessingWorker()
        w._sim_dt = 1.0 / 30.0
        w._sim_time = 0.0
        return w

    def _trk(self, locked=True, unc=2.0, resid=1.0, miss_dt=0.0,
             pred_only=False, state="TRACKING"):
        from types import SimpleNamespace
        return SimpleNamespace(
            state=SimpleNamespace(name=state), locked=locked,
            uncertainty_x=unc / 2.0, uncertainty_y=unc / 2.0,
            residual_magnitude=resid, prediction_only=pred_only,
            time_since_last_detection_s=miss_dt)

    def test_stable_tracking_arms_handoff_with_log(self, qapp):
        w = self._worker(qapp)
        msgs: list[str] = []
        w.log.connect(lambda level, msg: msgs.append(msg))
        for _ in range(15):
            w._sim_time += 1.0 / 30.0
            w._update_handoff(self._trk())
            qapp.processEvents()
        assert w._state.handoff_ready is True
        assert w._state.handoff_stable_s > 0.4
        assert any("HANDOFF READY" in m for m in msgs)

    def test_instability_clears_handoff(self, qapp):
        w = self._worker(qapp)
        for _ in range(15):
            w._sim_time += 1.0 / 30.0
            w._update_handoff(self._trk())
        assert w._state.handoff_ready is True
        w._sim_time += 1.0 / 30.0
        w._update_handoff(self._trk(locked=False))
        assert w._state.handoff_ready is False
        assert w._state.handoff_stable_s == 0.0

    def test_noisy_track_never_arms(self, qapp):
        w = self._worker(qapp)
        for _ in range(60):
            w._sim_time += 1.0 / 30.0
            w._update_handoff(self._trk(unc=60.0, resid=12.0))
        assert w._state.handoff_ready is False
