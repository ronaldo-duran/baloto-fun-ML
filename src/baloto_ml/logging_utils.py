"""Configuración de logging común para CLI, pipeline y notebooks."""

from __future__ import annotations

import contextlib
import logging
import sys

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_logging(level: str | int = "INFO") -> None:
    """Configura el logging raíz; tolera consolas de Windows sin UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):  # stream ya envuelto o sin soporte
                reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S", force=True)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
