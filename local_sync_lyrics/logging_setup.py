"""Application logging.

* A rotating UTF-8 log file in the app-data ``logs`` directory (always).
* A ``rich`` console handler only when a real terminal is attached.
* ``ensure_std_streams`` guards against ``sys.stdout``/``sys.stderr`` being
  ``None`` - which is the case in a PyInstaller ``--noconsole`` build and makes
  libraries that print or use tqdm (huggingface_hub!) crash.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys

from .app_paths import logs_dir

_CONFIGURED = False


def ensure_std_streams() -> None:
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the ``local_sync_lyrics`` logger once and return it."""
    global _CONFIGURED
    log = logging.getLogger("local_sync_lyrics")
    if _CONFIGURED:
        return log
    ensure_std_streams()
    log.setLevel(level)
    log.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            logs_dir() / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        log.addHandler(file_handler)
    except OSError:
        pass  # logging must never stop the app from starting

    try:
        if sys.stderr is not None and sys.stderr.isatty():
            from rich.logging import RichHandler

            console = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
            console.setFormatter(logging.Formatter("%(message)s"))
            log.addHandler(console)
    except Exception:  # rich missing or terminal oddities
        pass

    # Third-party chatter we do not want in the log file at INFO.
    for noisy in ("urllib3", "httpx", "httpcore", "huggingface_hub", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return log
