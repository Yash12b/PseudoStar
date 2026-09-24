"""Load an external video through the exact GUI worker path and score it.

Mirrors MainWindow._open_video_file -> _start_worker(mode=video):
fresh ProcessingWorker, no prior knowledge of the file, tracker starts
from NO_TRACK. Steps every frame synchronously and reports per-segment
detection rate, lock rate, acquisitions and losses.
"""
import sys
import time

sys.path.insert(0, "src")

SEG_S = 15
SEG_NAMES = [
    "BASE", "SIZE-5", "SIZE-20", "SHAPE-circ", "SHAPE-spot",
    "MOT-circ", "MOT-fig8", "MOT-rand", "MOT-spiral", "MOT-sinus",
    "MULTI-x3", "NOISE-gauss20", "NOISE-sp10", "NOISE-pois+lowlight",
    "FOG+RAIN+JIT20", "FOV-EXIT",
]


def main(path: str, backend: str = "classical") -> int:
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication([])

    from fsoc_tracker.gui.worker import ProcessingWorker
    w = ProcessingWorker()
    # Same dict _open_video_file passes (no GT, no hints about content).
    w.configure({"mode": "video", "video_path": path,
                 "perception_backend": backend})
    w.start_run()
    w._paused = True  # drive frames manually below
    app.processEvents()

    assert w._detector is not None
    rec = []  # (video_frame_index, detected, locked, state)
    t0 = time.perf_counter()
    while True:
        if w._source is None:
            break
        w._run_one_step()
        app.processEvents()
        s = w._state
        rec.append((w._frame_index, bool(s.perception.detected),
                    bool(s.tracking.locked and s.tracking.state == "TRACKING"),
                    s.tracking.state))
        if w._source is None or len(rec) >= 7300 \
                or (time.perf_counter() - t0) > 900:
            break
    wall = time.perf_counter() - t0
    n = len(rec)
    print(f"entries={n} last_video_frame={rec[-1][0] if rec else None} "
          f"wall={wall:.1f}s")
    print(f"{'seg':<18}{'frames':>7}{'det%':>7}{'lock%':>7}{'acq':>5}{'loss':>5}")
    bad = 0
    for si, name in enumerate(SEG_NAMES):
        idx = [i for i in range(n)
               if si * SEG_S * 30 <= rec[i][0] < (si + 1) * SEG_S * 30]
        if not idx:
            print(f"{name:<18}{'--':>7}")
            continue
        det = sum(rec[i][1] for i in idx) / len(idx) * 100
        lok = sum(rec[i][2] for i in idx) / len(idx) * 100
        sts = [rec[i][3] for i in idx]
        acq = sum(1 for k in range(1, len(sts))
                  if sts[k] == "TRACKING" and sts[k - 1] != "TRACKING")
        loss = sum(1 for k in range(1, len(sts))
                   if sts[k] in ("LOST", "SEARCHING", "REACQUIRING")
                   and sts[k - 1] == "TRACKING")
        flag = ""
        if si == 15:
            # FOV-EXIT by design: beacon mostly absent. Correct behavior
            # is a recorded loss with no static-text lock. WEAK only if
            # the tracker never registers the loss (text lock) or misses
            # the frames where the beacon is actually present.
            if loss < 1:
                flag = "  <-- WEAK (no loss on empty FOV)"
                bad += 1
        elif det < 80 or lok < 70:
            flag = "  <-- WEAK"
            bad += 1
        print(f"{name:<18}{len(idx):>7}{det:>6.1f}{lok:>6.1f}{acq:>5d}{loss:>5d}{flag}")
    try:
        w.stop_run()
    except Exception:
        pass
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "classical"))
