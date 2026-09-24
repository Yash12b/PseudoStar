"""Structured logging infrastructure for FSOC Tracker.

Provides:
    - ``setup_logging``: configure console and file handlers.
    - ``get_logger``: retrieve a namespaced logger.
    - Session-aware log file naming.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
)
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(
    log_level: str = "INFO",
    log_dir: str | Path = "logs",
    session_id: str = "",
    stream: TextIO | None = None,
) -> None:
    """Configure application-wide logging.

    Args:
        log_level: Minimum severity to emit (DEBUG, INFO, WARNING, ERROR).
        log_dir: Directory for log files.
        session_id: Session identifier used in the log filename.
        stream: Override stream for console handler (useful in tests).
    """
    global _configured  # noqa: PLW0603
    if _configured:
        return

    level = getattr(logging, log_level.upper(), logging.INFO)
    root = logging.getLogger("fsoc_tracker")
    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)

    # Console handler
    console = logging.StreamHandler(stream or sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler (if log_dir is writable)
    log_path = Path(log_dir)
    try:
        log_path.mkdir(parents=True, exist_ok=True)
        filename = f"{session_id}.log" if session_id else "fsoc_tracker.log"
        file_handler = logging.FileHandler(log_path / filename, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        root.warning("Could not create log file in %s", log_dir)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the ``fsoc_tracker`` hierarchy.

    Args:
        name: Module or component name (e.g. ``"core.frame"``).

    Returns:
        A ``logging.Logger`` instance.
    """
    return logging.getLogger(f"fsoc_tracker.{name}")
