"""SIH scorecard widget — pass/fail indicators."""

from __future__ import annotations

from fsoc_tracker.gui.state import SIHScorecard
from fsoc_tracker.gui.theme import Colors

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QFrame,
        QGridLayout,
        QLabel,
        QVBoxLayout,
    )
except ImportError:
    from PyQt5.QtCore import Qt  # type: ignore
    from PyQt5.QtWidgets import (
        QFrame,  # type: ignore
        QGridLayout,
        QLabel,
        QVBoxLayout,
    )


class ScorecardWidget(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._items: dict[str, dict] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        title = QLabel("SIH SCORECARD")
        title.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 11px; font-weight: bold; background: transparent; border: none;")
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setSpacing(3)

        metrics = [
            ("acquisition", "Acquisition", "s", 2.0),
            ("rmse", "RMSE", "px", 10.0),
            ("loss", "Loss Rate", "%", 5.0),
            ("reacq", "Re-acquisition", "s", 1.0),
            ("fps", "Processing FPS", "fps", 20.0),
        ]

        for i, (key, label, unit, threshold) in enumerate(metrics):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {Colors.MUTED}; font-size: 9px; background: transparent; border: none;")
            lbl.setFixedWidth(90)

            val_lbl = QLabel("--")
            val_lbl.setStyleSheet(f"color: {Colors.TEXT}; font-size: 10px; font-weight: bold; background: transparent; border: none;")
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val_lbl.setFixedWidth(70)

            status_lbl = QLabel("NOT EVAL")
            status_lbl.setStyleSheet(f"color: {Colors.MUTED}; font-size: 9px; font-weight: bold; background: transparent; border: none;")
            status_lbl.setAlignment(Qt.AlignCenter)
            status_lbl.setFixedWidth(60)

            grid.addWidget(lbl, i, 0)
            grid.addWidget(val_lbl, i, 1)
            grid.addWidget(status_lbl, i, 2)

            self._items[key] = {
                "value": val_lbl,
                "status": status_lbl,
                "unit": unit,
                "threshold": threshold,
            }

        layout.addLayout(grid)

        self._overall = QLabel("OVERALL: --")
        self._overall.setStyleSheet(f"color: {Colors.MUTED}; font-size: 11px; font-weight: bold; background: transparent; border: none; padding-top: 6px; border-top: 1px solid {Colors.PANEL_BORDER};")
        layout.addWidget(self._overall)
        layout.addStretch()

    _FIELD_MAP = {
        "acquisition": "acquisition_s",
        "rmse": "rmse_px",
        "loss": "loss_percent",
        "reacq": "reacq_s",
        "fps": "fps",
    }

    def update_state(self, scorecard: SIHScorecard) -> None:
        results = {}
        for key, item in self._items.items():
            field = self._FIELD_MAP.get(key, key)
            val = getattr(scorecard, field, None)
            threshold = item["threshold"]
            unit = item["unit"]

            if val is not None:
                item["value"].setText(f"{val:.2f} {unit}")
                if key == "rmse" or key == "loss":
                    passed = val <= threshold
                elif key == "fps":
                    passed = val >= threshold
                elif key == "acquisition" or key == "reacq":
                    passed = val <= threshold
                else:
                    passed = False

                if passed:
                    item["status"].setText("PASS")
                    item["status"].setStyleSheet(f"color: {Colors.SCORECARD_PASS}; font-size: 9px; font-weight: bold; background: transparent; border: none;")
                    results[key] = True
                else:
                    item["status"].setText("FAIL")
                    item["status"].setStyleSheet(f"color: {Colors.SCORECARD_FAIL}; font-size: 9px; font-weight: bold; background: transparent; border: none;")
                    results[key] = False
            else:
                item["value"].setText("--")
                item["status"].setText("NOT EVAL")
                item["status"].setStyleSheet(f"color: {Colors.SCORECARD_NA}; font-size: 9px; font-weight: bold; background: transparent; border: none;")

        if results:
            all_pass = all(results.values())
            n_pass = sum(1 for v in results.values() if v)
            n_total = len(results)
            if all_pass:
                self._overall.setText(f"OVERALL: PASS ({n_pass}/{n_total})")
                self._overall.setStyleSheet(f"color: {Colors.SCORECARD_PASS}; font-size: 11px; font-weight: bold; background: transparent; border: none; padding-top: 6px; border-top: 1px solid {Colors.PANEL_BORDER};")
            else:
                self._overall.setText(f"OVERALL: FAIL ({n_pass}/{n_total})")
                self._overall.setStyleSheet(f"color: {Colors.SCORECARD_FAIL}; font-size: 11px; font-weight: bold; background: transparent; border: none; padding-top: 6px; border-top: 1px solid {Colors.PANEL_BORDER};")
        else:
            self._overall.setText("OVERALL: NOT EVALUATED")
            self._overall.setStyleSheet(f"color: {Colors.MUTED}; font-size: 11px; font-weight: bold; background: transparent; border: none; padding-top: 6px; border-top: 1px solid {Colors.PANEL_BORDER};")
