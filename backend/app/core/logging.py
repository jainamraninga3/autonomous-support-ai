"""Structured logging configuration."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import get_settings

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_FILE = LOG_DIR / "app.log"


def configure_logging() -> None:
    """Configure application-wide structured logging.

    Uses a simple structured (key=value) text format for now. This keeps
    Phase 1 dependency-free while remaining easy to swap for JSON logging
    later without touching call sites.

    Logs to stdout (as before) AND to a rotating file under `backend/
    logs/app.log` (5MB x 3 backups) — added specifically so a 500 error's
    real traceback (logged via `logger.exception(...)` in
    `app.core.exceptions`) survives after the terminal that printed it is
    gone, e.g. for post-mortem on a test run's failures.
    """
    settings = get_settings()

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s level=%(levelname)s logger=%(name)s "
            "message=%(message)s"
        ),
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.LOG_LEVEL.upper())
    root_logger.handlers = [stream_handler, file_handler]

    # Keep noisy third-party loggers at a sane level.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
