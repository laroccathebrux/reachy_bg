"""Logging helpers: one call to configure, ``get_logger`` everywhere else."""

from __future__ import annotations

import logging
import sys

from src.config import LOG_FILE, LOG_LEVEL

_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"
_configured = False


def configure_logging(level: str = LOG_LEVEL, *, to_file: bool = True) -> None:
    """Configure the root logger once (stdout, optionally the app log file)."""
    global _configured
    if _configured:
        return
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if to_file:
        handlers.append(logging.FileHandler(LOG_FILE, encoding="utf-8"))
    logging.basicConfig(level=level, format=_FORMAT, handlers=handlers)
    # Third-party HTTP clients log every request at INFO; keep them quiet unless debugging.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Logger for a module; configures logging on first use."""
    configure_logging()
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
