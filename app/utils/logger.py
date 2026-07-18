"""
app/utils/logger.py
─────────────────────────────────────────────────────────────────────────────
Structured logging configuration for the Rossmann Forecasting API.

Usage in any module:
    from app.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened", extra={"store_id": 42})

Log level is controlled via the LOG_LEVEL environment variable
(see app/config.py) — defaults to INFO if unset or invalid.

The file handler is a RotatingFileHandler rather than a plain FileHandler:
a long-running production container would otherwise accumulate an
unbounded rossmann_api.log. Rotation caps disk usage at
MAX_LOG_BYTES * (BACKUP_COUNT + 1).
"""

import logging
from logging.handlers import RotatingFileHandler
import sys

from app.config import LOGS_DIR, LOG_LEVEL_NAME


LOG_FILE   = LOGS_DIR / "rossmann_api.log"
LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.INFO)

MAX_LOG_BYTES = 10 * 1024 * 1024   # 10 MB per file
BACKUP_COUNT  = 5                  # keep 5 rotated backups (60 MB max on disk)

_configured = False


def _configure_root_logger() -> None:
    """Configure the root logger once on first import."""
    global _configured
    if _configured:
        return

    formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(formatter)

    # Rotating file handler — bounded disk usage instead of an ever-growing log file
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger. Configures the root logger on first call.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance.
    """
    _configure_root_logger()
    return logging.getLogger(name)