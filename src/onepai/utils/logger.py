"""Logger strutturato per ONEPAI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

__all__ = ["get_logger", "configure_logging"]

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Configura il logger root di ONEPAI una volta sola.
    level: "DEBUG" | "INFO" | "WARNING" | "ERROR"
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    root = logging.getLogger("onepai")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    # Console
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(_FORMATTER)
    root.addHandler(ch)
    # File opzionale
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(_FORMATTER)
        root.addHandler(fh)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Ritorna un logger figlio di 'onepai'.
    Se la configurazione non è stata fatta esplicitamente, usa INFO sul stderr.
    """
    configure_logging()
    return logging.getLogger(f"onepai.{name}")
